# Context-Sensitive Adaptive Retrieval в multi-turn RAG

НИР, ИТМО, 2026/27. В многоходовом диалоговом RAG роутер по запросу и истории диалога выбирает стратегию поиска: как сформировать запрос (переписывать ли его), какой ретривер использовать и нужен ли reranking. Сравниваются три режима: **фиксированный пайплайн**, **адаптация без учёта истории**, **адаптация с учётом истории** — по качеству поиска, качеству ответов и стоимости.

- **RQ1.** Есть ли смысл выбирать стратегию для каждого запроса? Сравниваем оракул с лучшей фиксированной стратегией.
- **RQ2.** Помогает ли роутеру история диалога?
- **RQ3.** Какой компромисс «качество / стоимость» дают адаптивные роутеры?

Подробности: [журнал исследования](docs/journal.md) · [методика и определения](docs/methodology.md) · [результаты](docs/results/) · [сводка](docs/README.md).

## Статус по плану

| Этап | Статус |
|---|---|
| 2. Постановка, датасеты, метрики | ✅ MTRAG; ⏳ MTRAG-UN (загрузчик есть, прогонов нет) |
| 3. Бейзлайны и экспериментальная среда | ✅ BM25, dense, hybrid + reranker, LLM query rewriting, CHIQ, Adaptive-RAG; LLM лаборатории; Recall/nDCG, latency, токены; LLM-as-a-Judge с калибровкой на оценках людей |
| 4. Роутер и сравнение трёх режимов | ✅ rule-based и LLM-based роутеры; сравнение по поиску, ответам и стоимости |
| 5. Отчётные материалы | ⏳ презентация |

## Главные результаты (MTRAG)

Срез — последующие ходы диалога (n = 675), где история влияет на запрос. Метрики: nDCG@5 поиска и correctness ответа (LLM-судья, шкала 1–5). LLM-токены системы — переписывание запроса, роутер и генерация ответа.

| режим | система | nDCG@5 | correctness | LLM-токены системы |
|---|---|---|---|---|
| фиксированный | `last:bm25` | 0.176 | — | — |
| фиксированный | `last:dense` | 0.307 | 3.22 | 1726 |
| фиксированный (из плана) | `last:hybrid+rerank` | 0.351 | 3.27 | 1710 |
| фиксированный, бейзлайн CHIQ | `chiq:dense+rerank` | 0.405 | 3.51 | 3948 |
| **фиксированный, лучший** | `llm_rewrite:dense+rerank` | **0.416** | **3.53** | 2176 |
| без истории | `llm_nohist` (LLM-роутер) | 0.313 | 3.19 | 1912 |
| без истории | `adaptive_rag` (бейзлайн Adaptive-RAG) | 0.394 | 3.44 | 1896 |
| с историей | `llm_hist` (LLM-роутер) | 0.352 | 3.33 | 2581 |
| с историей | `rule` (правила) | 0.393 | 3.49 | 1997 |
| верхняя граница | `oracle` | 0.619 | 3.69 | 1873 |

- **Dense (`bge-m3`) намного сильнее BM25; reranker и переписывание запроса через LLM значимо помогают и складываются.** LLM-переписывание значимо лучше человеческого rewrite из MTRAG.
- **CHIQ** с dense и reranker не лучше простого LLM-переписывания, а стоит в 5 раз больше токенов на этапе поиска. Зато он сильно помогает BM25 и hybrid за счёт расширения запроса и выигрывает при смене темы.
- **RQ1: запас для адаптации большой.** Оракул даёт 0.619 против 0.416 по nDCG@5. При этом LLM-переписывание нужно ему только в ~25% запросов, а reranker — в ~32%.
- **RQ3:** `rule` и `adaptive_rag` по качеству ответов статистически неотличимы от лучшего фиксированного пайплайна, но дешевле на 8% и 13% LLM-токенов системы. `adaptive_rag` к тому же вызывает reranker только в 59% случаев.
- **LLM-роутеры проигрывают.** Решение «переписывать ли» LLM принимает хорошо, а ретривер и reranker выбирает плохо. Вызов LLM-роутера с историей (623 токена) дороже самого переписывания (428).
- **RQ2:** история нужна для самого переписывания. Но решение «на каком уровне обрабатывать вопрос» обученный классификатор без истории (`adaptive_rag`) принимает не хуже роутеров с историей.
- **Улучшения поиска переходят в качество ответов** (ранговая корреляция систем ρ = 0.97), но разрыв на ответах в разы меньше: узкое место смещается в генерацию.
- **Судья `qwen3.8`** согласуется с оценками людей на уровне ρ ≈ 0.40–0.63 (477 ответов MTRAG). Это не хуже автоматических метрик авторов MTRAG.

## Установка

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -e ".[neural,dev]"     # [neural] — torch и sentence-transformers для локального reranker
./scripts/download_data.sh         # MTRAG + MTRAG-UN в data/mt-rag-benchmark (~530 МБ)
cp .env.example .env               # вписать LLM_BASE_URL и LLM_API_KEY
source .env
arag llm-check                     # API, модели, эмбеддинги bge-m3
arag rerank-check                  # локальный reranker и его скорость
pytest -q
```

Модели лаборатории подключаются через LiteLLM (OpenAI-совместимый API). Имена моделей и параметры (таймаут, повторы, `workers`, отключение thinking у Qwen) задаются в [config.yaml](config.yaml).

| роль | модель | где работает |
|---|---|---|
| переписывание запроса, CHIQ, LLM-роутеры, судья | `qwen3.8` | API |
| генерация ответа | `gemma-4-26B-A4B-it` | API |
| dense-эмбеддинги | `bge-m3` | API (`/v1/embeddings`) |
| reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | локально (`mps` / `cpu`) |

## Данные

| Датасет | Задачи | С qrels | Домены |
|---|---|---|---|
| `mtrag` (MTRAG human) | 842 | 777 | clapnq, cloud, fiqa, govt |
| `mtragun` (MTRAG-UN) | 507 | 332 | те же; Banking и Telco ещё не опубликованы |

Корпуса passage-level (512 токенов, overlap 100, 366 тыс. пассажей), id совпадают с qrels. Задача хранит всю историю (`turns`), human rewrite (`gold_rewrite`), эталонный ответ и метки: answerability, multi-turn, тип вопроса.

## Стратегии и роутеры

**Стратегия** — строка `<query>:<retriever>[+rerank]`, всего 30 (`all`).

| часть | варианты |
|---|---|
| query | `last` — последний вопрос · `user_concat` — 3 последние реплики пользователя · `gold_rewrite` — human rewrite (на инференсе недоступен) · `llm_rewrite` — переписывание LLM, 1 вызов · `chiq` — CHIQ-AD, 5–6 вызовов |
| retriever | `bm25` · `dense` (`bge-m3`) · `hybrid` (RRF) |
| `+rerank` | cross-encoder по top-50 кандидатам |

Группы для `--strategies`: `all` (30), `no_llm` (18), `no_rerank` (15).

**Роутеры** (`arag route --router ...`) выбирают стратегию для каждой задачи:

| роутер | что видит | что решает |
|---|---|---|
| `fixed --fixed <стратегия>` | — | одна стратегия для всех |
| `rule` | вопрос, есть ли история | переписывать только зависимые от контекста вопросы; всегда dense + reranker (`config.yaml` → `routers.rule`) |
| `llm_nohist` | только вопрос | ретривер и reranker |
| `llm_hist` | вопрос и историю | переписывание, ретривер и reranker |
| `adaptive_rag` | только вопрос | уровень сложности A / B / C → `last:dense` / `last:dense+rerank` / `llm_rewrite:dense+rerank`; классификатор на эмбеддингах, кросс-валидация по диалогам |
| `oracle`, `oracle_cheap` | разметку оракула | лучшая / самая дешёвая из почти лучших стратегий (верхняя граница) |

## Полный прогон

```bash
caffeinate -i arag index                                  # dense-индексы через API (~12 мин, с докачкой)
caffeinate -i arag run --strategies all                   # 30 стратегий; готовые пропускаются
arag eval --slice later_turns                             # Recall/nDCG, latency, токены по стратегиям
arag oracle                                               # оракул и разметка
arag analyze --out docs/results/mtrag_analysis.md         # отчёт по срезам
arag significance --baseline last:dense --slice later_turns

for r in rule llm_nohist llm_hist adaptive_rag oracle oracle_cheap; do arag route --router $r; done
arag route --router fixed --fixed llm_rewrite:dense+rerank
arag routes-summary --slice later_turns                   # роутеры: качество, решения, стоимость

arag judge-calibrate --out docs/results/judge_calibration.md   # согласие судьи с оценками людей
arag answer --run "runs/mtrag/llm_rewrite__dense+rerank.jsonl" runs/mtrag_routes/rule.jsonl   # ответы + судья
arag answers-summary --slice later_turns --baseline "llm_rewrite:dense+rerank"
```

Все команды принимают `--dataset mtrag|mtragun`, `--domains`, `--limit`. Команды с LLM дополнительно принимают `--workers`.

## Команды

| команда | что делает | куда пишет |
|---|---|---|
| `llm-check` | проверка API, моделей, эмбеддингов | — |
| `rerank-check` | проверка reranker, замер скорости | — |
| `index` | dense-индексы корпусов через API | `cache/index/<домен>/` |
| `run` | прогон стратегий | `runs/<ds>/<стратегия>.jsonl` |
| `eval` | метрики по стратегиям, срезы и домены | терминал, `runs/<ds>_summary.csv` |
| `oracle` | оракул: реализуемый, cost-aware (`--eps`), с gold | `runs/<ds>_oracle.jsonl` |
| `analyze` | отчёт: датасет, срезы, оракул, когда помогает rewrite | `--out` |
| `significance` | парный bootstrap, рандомизационный тест, поправка Холма | терминал |
| `route` | роутер поверх готовых прогонов | `runs/<ds>_routes/<роутер>.jsonl` |
| `routes-summary` | сводка роутеров: качество, решения, стоимость | терминал, CSV |
| `judge-calibrate` | судья на 477 ответах MTRAG с оценками людей | `--out` (.md + .jsonl) |
| `answer` | генерация ответа и оценка судьёй (`--sample` — одна выборка для всех систем) | `runs/<ds>_answers/` |
| `answers-summary` | таблица качества ответов и значимость (`--baseline`) | терминал, CSV |

**Кэш.** Ответы LLM, эмбеддинги запросов и оценки пар reranker кэшируются в `cache/`. Упавший или прерванный прогон при перезапуске продолжается с места остановки. Latency при попадании в кэш берётся из первого вызова. Для честных замеров latency LLM-этапов запускать с `--workers 1`. `--limit` пишет усечённый run-файл; для полного прогона перезапустить с `--overwrite`.

## Метрики

- **Поиск:** Recall@{1,3,5,10}, nDCG@{1,3,5,10} (pytrec_eval, как в официальном скрипте MTRAG). Основная — nDCG@5.
- **Стоимость:** LLM-токены, число пар reranker, latency по этапам (переписывание, эмбеддинг запроса, поиск, rerank, роутер).
- **Ответы:** correctness, faithfulness, completeness (1–5) от LLM-судьи. Калибровка — на human-оценках MTRAG.
- **Значимость:** парный bootstrap (95% ДИ) и рандомизационный тест с поправкой Холма.

## Ограничения

- **Reranker — лёгкая MiniLM:** на M1 Air 8 ГБ `bge-reranker-v2-m3` слишком медленный. Latency reranker на ноутбуке нестабильна (троттлинг), поэтому стоимость сравниваем по токенам и числу пар.
- **Ответы агента в истории MTRAG — эталонные.** LLM-переписывание может частично выигрывать за счёт этого.
- **Оцениваются только задачи с qrels**; неотвечаемые вопросы не оцениваются.
- **CHIQ:** реализован только ad-hoc вариант (CHIQ-AD). **Adaptive-RAG:** классификатор — логистическая регрессия на эмбеддингах вместо T5-large. Отличия от оригиналов — в [методике](docs/methodology.md).
- **Правила `rule` выведены из анализа тех же данных MTRAG.** Проверка на отложенных данных — MTRAG-UN.

## Структура репозитория

```
src/arag/
  cli.py          команды arag
  data.py         загрузка задач, корпусов, qrels
  queries.py      формирование запроса: last, user_concat, gold_rewrite, llm_rewrite, chiq
  retrievers.py   BM25, dense (локальный и через API), RRF, reranker с кэшем пар
  pipeline.py     стратегия = запрос + ретривер + rerank; Engine выполняет её на задаче
  routers.py      fixed, rule, llm_nohist, llm_hist, adaptive_rag, oracle
  generation.py   генерация ответа и LLM-судья
  judging.py      калибровка судьи, сводка качества ответов
  llm.py          OpenAI-совместимый клиент: кэш, повторы, параллельность, эмбеддинги
  metrics.py, stats.py, analysis.py   метрики, тесты значимости, отчёты
docs/             журнал, методика, отчёты с результатами
tests/            юнит-тесты на фейковых LLM и моделях
config.yaml       модели, параметры поиска, роутеров, API
```

`data/`, `runs/`, `cache/` в git не попадают. Результаты, которые нужно сохранить, лежат в `docs/results/` и в журнале; логи терминала — в `runs/logs/`.
