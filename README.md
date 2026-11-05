# Context-Sensitive Adaptive Retrieval в multi-turn RAG

НИР, ИТМО, 2026/27. Роутер анализирует запрос и историю диалога и выбирает стратегию retrieval: формирование запроса, ретривер, reranking. Сравниваем три режима: фиксированный пайплайн, адаптацию без истории, адаптацию с историей.

## Установка

```bash
pip install -e ".[neural,dev]"     # без [neural] работает только BM25
./scripts/download_data.sh         # MTRAG + MTRAG-UN в data/mt-rag-benchmark
cp .env.example .env && source .env
```

Модели на сервере лаборатории подключаются через OpenAI-совместимый API: `LLM_BASE_URL`, `LLM_API_KEY`. Имена моделей в `config.yaml`.

## Данные

| Датасет | Задачи | С qrels | Домены |
|---|---|---|---|
| `mtrag` (MTRAG human) | 842 | 777 | clapnq, cloud, fiqa, govt |
| `mtragun` (MTRAG-UN) | 507 | 332 | те же; Banking и Telco ещё не опубликованы |

Корпуса passage-level (512 токенов, overlap 100), id совпадают с qrels. Задача хранит всю историю (`turns`), human-rewrite (`gold_rewrite`) и метки: answerability, multi_turn, question_type, эталонный ответ.

## Стратегия

Строка вида `<query>:<retriever>[+rerank]`:

- query: `last` · `user_concat` (последние 3 реплики пользователя) · `gold_rewrite` (human rewrite, верхняя граница для rewrite) · `llm_rewrite`
- retriever: `bm25` · `dense` · `hybrid` (RRF)
- `+rerank`: cross-encoder по top-50

Всего 24 стратегии (`all`), из них 12 без LLM (`no_llm`).

## Команды

```bash
arag run --strategies no_llm                          # бейзлайны без LLM
arag run --strategies llm_rewrite:hybrid+rerank       # с LLM-rewrite
arag run --dataset mtragun --strategies all

arag eval --slice later_turns                         # таблица метрик, срезы all / first_turn / later_turns
arag eval --by-domain

arag oracle --eps 0.05                                # лучшая стратегия на каждый запрос → разметка для роутера
arag significance --baseline last:bm25 --slice later_turns   # парный bootstrap + рандомизационный тест, поправка Холма

arag route --router fixed --fixed last:hybrid+rerank  # фиксированный пайплайн
arag route --router llm_nohist                        # адаптация без истории (Adaptive-RAG-like)
arag route --router llm_hist                          # адаптация с историей
arag route --router rule

arag answer --run runs/mtrag/llm_rewrite__hybrid+rerank.jsonl --limit 100   # генерация + LLM-as-a-Judge
```

`oracle` считает три варианта: реализуемый (без `gold_rewrite`, это и есть разметка `best`), cost-aware (самая дешёвая стратегия в пределах `eps` от лучшей, `best_cheap`) и с `gold_rewrite` как верхнюю границу (`best_with_gold`).

`significance` сравнивает с baseline все runs из `runs/<dataset>/` и роутеры из `runs/<dataset>_routes/` (имя `route/<router>`).

`route` берёт результаты из уже посчитанных runs, если стратегия там есть; иначе выполняет её на лету. Поэтому сначала стоит прогнать `run --strategies all`, тогда сравнение роутеров почти бесплатно.

Флаг `--limit` пишет усечённый run-файл. Для полного прогона перезапустить с `--overwrite`.

LLM-вызовы кэшируются в `cache/llm_cache.sqlite`. Latency при попадании в кэш берётся из первого вызова.

## Метрики

- Retrieval: Recall@{1,3,5,10}, nDCG@{1,3,5,10} (pytrec_eval, как в официальном скрипте MTRAG)
- Стоимость: latency по этапам (rewrite, retrieve, rerank, router), LLM-токены, число пар для reranker
- Ответы: correctness, faithfulness, completeness (1–5), судья задаётся `llm.judge_model`

## Бейзлайны из обсуждения

| Бейзлайн | Как получить |
|---|---|
| BM25, dense | `last:bm25`, `last:dense` |
| Fixed Hybrid RAG + Reranker | `last:hybrid+rerank`, `llm_rewrite:hybrid+rerank` |
| Query rewriting по истории | `llm_rewrite:*`; `gold_rewrite:*` как потолок |
| CHIQ | TODO: отдельный `query mode` с history enhancement |
| SemEval-2026 Task 8 | TODO: разобрать решения участников |
| Adaptive-RAG | `route --router llm_nohist`; TODO: обучаемый классификатор сложности |

## Первые результаты (BM25, MTRAG, fiqa + govt, later turns, n=332)

| Стратегия | R@5 | R@10 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| gold_rewrite:bm25 | 0.225 | 0.317 | 0.196 | 0.236 |
| last:bm25 | 0.185 | 0.247 | 0.171 | 0.197 |
| user_concat:bm25 | 0.174 | 0.249 | 0.156 | 0.188 |

Оракул по трём BM25-стратегиям: nDCG@5 0.293 против 0.216 у лучшей фиксированной. Запас для роутера есть уже на уровне формирования запроса.

## TODO

- [ ] Dense и reranker на GPU, все 24 стратегии на 4 доменах
- [ ] Подключить Qwen / gemma, прогнать `llm_rewrite`
- [ ] CHIQ, решения SemEval-2026 Task 8
- [ ] Обучаемый роутер на разметке оракула (cost-aware)
- [ ] Калибровка судьи на `reference_subset_with_human_evaluations.json`
- [ ] Pareto «качество vs стоимость»
