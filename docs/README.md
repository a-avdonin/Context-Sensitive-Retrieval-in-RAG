# Документация НИР

| Файл | Что внутри |
|---|---|
| [journal.md](journal.md) | Журнал исследования: что сделано, зачем, результаты, решения, что дальше. Новые записи сверху. |
| [methodology.md](methodology.md) | Определения: данные, стратегии, оракулы, метрики, срезы, тесты значимости. Сюда смотреть, если непонятно, что значит цифра. |
| [results/](results/) | Отчёты по поиску `arag analyze`: `mtrag_bm25_analysis.md` (3 стратегии), `mtrag_norerank_analysis.md` (12), `mtrag_rerank_analysis.md` (24, без CHIQ). Качество ответов: `mtrag_answers.md`. Калибровка судьи: `judge_calibration_qwen.md` (v1), `judge_calibration_qwen_v2.md` (актуальная). |

## Где мы сейчас (16.11.2026)

Этапы 3 и 4 плана закрыты. Осталось: MTRAG-UN (этап 2) и презентация (этап 5). Краткая версия — в [README проекта](../README.md).

MTRAG, 777 задач с qrels; срез — последующие ходы (n = 675), метрика nDCG@5. Подробности и значимость — в журнале.

| стратегия | nDCG@5 | цена |
|---|---|---|
| `last:bm25` | 0.176 | ~0 |
| `last:bm25+rerank` | 0.300 | 50 пар reranker |
| `last:dense` (bge-m3) | 0.307 | эмбеддинг запроса |
| `gold_rewrite:dense` (human rewrite) | 0.336 | недоступна на инференсе |
| `last:dense+rerank` (MiniLM) | 0.349 | + 50 пар reranker |
| `llm_rewrite:dense` (qwen3.8) | 0.368 | + ~430 LLM-токенов |
| `chiq:hybrid` (CHIQ-AD, без reranker) | 0.383 | ~2180 LLM-токенов |
| `chiq:dense+rerank` (CHIQ-AD) | 0.405 | ~2180 LLM-токенов + 50 пар |
| **`llm_rewrite:dense+rerank`** — лучшая фиксированная | **0.416** | LLM + reranker |

Оракул по всем 24 стратегиям, все 777 задач: **0.631** против 0.443 у лучшей фиксированной (+42%). Из решаемых запросов LLM нужен только 25%, reranker — 37%, а 48% лучше всего обслуживаются без того и другого.

**Роутеры (этап 4)**, последующие ходы, nDCG@5:

| | nDCG@5 | LLM-токены |
|---|---|---|
| лучший фиксированный пайплайн | 0.416 | 428 |
| `adaptive_rag` (Adaptive-RAG, без истории) | 0.394 | 147 |
| `rule` (rewrite только для зависимых вопросов) | 0.393 | 251 |
| `llm_hist` (адаптация с историей) | 0.352 | 838 |
| `llm_nohist` (без истории) | 0.313 | 196 |
| оракул | 0.619 | 111 |

Adaptive-RAG даёт качество как у `rule` при в 1.7 раза меньших затратах и теряет 5% относительно лучшего фиксированного пайплайна, экономя 66% LLM-токенов. LLM-роутер хорошо решает, когда переписывать вопрос, но плохо выбирает ретривер и reranker. К тому же его вызов (623 токена) дороже самого переписывания (428), см. журнал.

**Качество ответов** (LLM-судья, later turns, шкала 1–5; [results/mtrag_answers.md](results/mtrag_answers.md)):

| | correctness | LLM-токены системы (поиск + генерация) |
|---|---|---|
| лучший фиксированный `llm_rewrite:dense+rerank` | 3.53 | 2176 |
| `rule` | 3.49 (н.з.) | 1997 (−8%) |
| `adaptive_rag` | 3.44 (н.з.) | 1896 (−13%) |
| `llm_hist` | 3.33 | 2581 (+19%) |
| `last:dense` | 3.22 | 1726 |
| оракул | 3.69 | 1873 |

Ранжирование систем по ответам совпадает с ранжированием по поиску (ρ = 0.97), но разрыв меньше: узкое место смещается в генерацию. Судья `qwen3.8` согласуется с людьми на уровне ρ ≈ 0.40–0.63 (калибровка на 477 ответах MTRAG).

**Ключевые выводы на сейчас:**
- dense намного сильнее BM25;
- переписывание через LLM значимо лучше человеческого rewrite;
- reranker значимо помогает;
- переписывание и reranker дают сопоставимый прирост и складываются;
- reranker вытягивает даже BM25 до уровня dense;
- CHIQ с dense и reranker не лучше простого LLM-переписывания (0.405 против 0.416, не значимо) при стоимости в 5 раз выше; но он сильно помогает BM25 и hybrid за счёт расширения запроса и выигрывает при смене темы;
- у адаптивного роутера большой запас по качеству и по экономии.

## Окружение

- MacBook Air M1, 8 ГБ. LLM и эмбеддинги через LiteLLM лаборатории (`.env`: `LLM_BASE_URL`, `LLM_API_KEY`). Reranker локально на `mps`.
- Модели: `qwen3.8` (rewrite, роутер, судья), `gemma-4-26B-A4B-it` (генерация), `bge-m3` (эмбеддинги), `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker).

## Как воспроизвести

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -e ".[neural,dev]"     # [neural] — torch для локального reranker
./scripts/download_data.sh         # ~530 МБ в data/
cp .env.example .env               # вписать ключ
source .env

arag llm-check                     # API, модели, bge-m3
arag rerank-check                  # reranker и его скорость
caffeinate -i arag index           # dense-индексы через API, ~12 мин, с докачкой
caffeinate -i arag run --strategies all    # 30 стратегий (вкл. CHIQ); готовые пропускаются, LLM/эмбеддинги/пары в кэше
arag eval --slice later_turns
arag oracle
arag significance --baseline last:dense --slice later_turns
arag analyze --out docs/results/mtrag_analysis.md
for r in rule llm_nohist llm_hist adaptive_rag oracle; do arag route --router $r; done
arag routes-summary --slice later_turns
arag judge-calibrate --out docs/results/judge_calibration_qwen_v2.md
arag answer --run "runs/mtrag/llm_rewrite__dense+rerank.jsonl" runs/mtrag_routes/rule.jsonl   # и др. системы
arag answers-summary --slice later_turns --baseline "llm_rewrite:dense+rerank"
pytest -q
```

`data/`, `runs/`, `cache/` в git не попадают. Все результаты, которые нужно сохранить, копируются в `docs/results/` или записываются в журнал. Логи терминала лежат в `runs/logs/`.
