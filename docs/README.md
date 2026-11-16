# Документация НИР

| Файл | Что внутри |
|---|---|
| [journal.md](journal.md) | Журнал исследования: что сделано, зачем, результаты, решения, что дальше. Новые записи сверху. |
| [methodology.md](methodology.md) | Определения: данные, стратегии, оракулы, метрики, срезы, тесты значимости. Сюда смотреть, если непонятно, что значит цифра. |
| [results/](results/) | Отчёты `arag analyze`: `mtrag_bm25_analysis.md` (3 стратегии), `mtrag_norerank_analysis.md` (12), `mtrag_rerank_analysis.md` (все 24, актуальный). |

## Где мы сейчас (16.11.2026)

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
| `rule` (rewrite только для зависимых вопросов) | 0.393 | 251 |
| `llm_hist` (адаптация с историей) | 0.352 | 838 |
| `llm_nohist` (без истории) | 0.313 | 196 |
| оракул | 0.619 | 111 |

LLM-роутер хорошо решает, когда переписывать вопрос, но плохо выбирает ретривер и reranker. К тому же его вызов (623 токена) дороже самого переписывания (428), см. журнал.

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
caffeinate -i arag run --strategies all    # 24 стратегии; готовые пропускаются, LLM/эмбеддинги/пары в кэше
arag eval --slice later_turns
arag oracle
arag significance --baseline last:dense --slice later_turns
arag analyze --out docs/results/mtrag_analysis.md
pytest -q
```

`data/`, `runs/`, `cache/` в git не попадают. Все результаты, которые нужно сохранить, копируются в `docs/results/` или записываются в журнал. Логи терминала лежат в `runs/logs/`.
