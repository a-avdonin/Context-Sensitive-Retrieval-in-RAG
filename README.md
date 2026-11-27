# Context-Sensitive Retrieval in Retrieval-Augmented Language Models

Индустриальный проект НИР ИТМО. Сравниваем фиксированный поиск с роутерами, которые для каждого вопроса выбирают способ формирования запроса, ретривер и необходимость реранкинга. Эксперименты проведены на MTRAG и MTRAG-UN.

## Основные результаты

Средние по последующим ходам с разметкой релевантности. Correctness оценивает LLM-судья по шкале 1–5. Токены системы включают переписывание, роутер и генерацию ответа.

| Датасет | Система | nDCG@5 | Correctness | LLM-токены на задачу |
|---|---|---:|---:|---:|
| MTRAG (675) | Фиксированный `llm_rewrite:dense+rerank` | 0.416 | 3.529 | 2175 |
| MTRAG (675) | `rule` | 0.393 | 3.493 | 1997 |
| MTRAG (675) | `adaptive_rag` | 0.394 | 3.436 | 1896 |
| MTRAG-UN (309) | Фиксированный `llm_rewrite:dense+rerank` | 0.4933 | 3.806 | 2083 |
| MTRAG-UN (309) | Фиксированный `llm_rewrite:hybrid+rerank` | 0.5093 | 3.841 | 2076 |
| MTRAG-UN (309) | `rule` | 0.4777 | 3.825 | 1916 |
| MTRAG-UN (309) | `adaptive_rag_from_mtrag` | 0.4566 | 3.657 | 1853 |

- **Запас качества.** Оракул на MTRAG получил 0.619 nDCG@5 при 0.416 у лучшего фиксированного пайплайна. [Отчёт по поиску](docs/results/mtrag_analysis.md)
- **Экономия.** `rule` расходует на 8% меньше токенов системы на обоих датасетах относительно фиксированного dense-пайплайна. Разница по correctness статистически незначима. [MTRAG](docs/results/mtrag_answers.md) · [MTRAG-UN](docs/results/mtragun.md)
- **История на входе роутера.** В контрольном сравнении на MTRAG-UN расход токенов вырос на 18%. Разницы по nDCG@5 (+0.0149) и correctness (+0.0453) статистически незначимы. [Контрольный эксперимент](docs/results/mtragun.md#контрольное-сравнение-для-rq2)

## Как повторить

Для прогона BM25 достаточно Python 3.10+ и данных. Лабораторный API для этого прогона не нужен. Команды выполняются из корня проекта.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e ".[dev]"
./scripts/download_data.sh
arag run --dataset mtrag --strategies last:bm25
arag eval --dataset mtrag --slice later_turns
```

Для результатов таблицы понадобятся зависимости `.[neural,dev]` и доступ к моделям лаборатории через LiteLLM. [Команды полных прогонов MTRAG и MTRAG-UN](docs/reproduction.md)

[Методика](docs/methodology.md) · [Калибровка LLM-судьи](docs/results/judge_calibration_qwen_v2.md) · [Все отчёты](docs/README.md)
