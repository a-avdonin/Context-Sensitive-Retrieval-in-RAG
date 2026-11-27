# Документация проекта

Таблицы по 13 системам на MTRAG и 14 на MTRAG-UN, сравнение доменов и выводы по исследовательским вопросам собраны в [README проекта](../README.md).

| Что нужно | Где искать |
|---|---|
| Запустить эксперименты с нуля или продолжить по сохранённым результатам | [Воспроизведение](reproduction.md) |
| Разобраться в датасетах, роутерах, метриках и правилах сравнения | [Методика](methodology.md) |
| Изучить поиск по доменам, типам вопросов и глубине диалога | [Анализ 30 стратегий на MTRAG](results/mtrag_analysis.md) |
| Проверить значимость различий и связь поиска с качеством ответа | [Парные сравнения на MTRAG](results/mtrag_answers.md) |
| Проверить вклад истории, перенос Adaptive-RAG и результаты CHIQ-AD | [Эксперименты на MTRAG-UN](results/mtragun.md) |
| Сопоставить оценки судьи с человеческими | [Калибровка v2](results/judge_calibration_qwen_v2.md) · [Оценки по ответам](results/judge_calibration_qwen_v2.jsonl) |

## Промежуточные отчёты

[BM25](results/mtrag_bm25_analysis.md) · [Поиск без реранкера](results/mtrag_norerank_analysis.md) · [Реранкинг до CHIQ-AD](results/mtrag_rerank_analysis.md) · [Калибровка судьи v1](results/judge_calibration_qwen.md)
