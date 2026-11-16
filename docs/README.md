# Документация НИР

| Файл | Что внутри |
|---|---|
| [journal.md](journal.md) | Журнал исследования: что сделано, зачем, результаты, решения, что дальше. Новые записи сверху. |
| [methodology.md](methodology.md) | Определения: данные, стратегии, оракулы, метрики, срезы, тесты значимости. Сюда смотреть, если непонятно, что значит цифра. |
| [results/](results/) | Сгенерированные отчёты (`arag analyze`) и таблицы, на которые ссылается журнал. |

## Как воспроизвести всё, что есть в журнале

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"            # без [neural]: только BM25, без torch
./scripts/download_data.sh         # ~530 МБ в data/

arag run --strategies last:bm25 user_concat:bm25 gold_rewrite:bm25   # ~1 мин на CPU
arag eval --slice later_turns
arag oracle
arag significance --baseline last:bm25
arag analyze --out docs/results/mtrag_bm25_analysis.md
pytest -q
```

`data/`, `runs/`, `cache/` в git не попадают. Всё, что нужно сохранить как результат, копируется в `docs/results/`.
