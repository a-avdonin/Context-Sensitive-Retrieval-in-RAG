# Воспроизведение экспериментов

## Подготовка окружения

Запускайте команды из корня репозитория с Python 3.10 или новее. Для прогона BM25 без лабораторного API достаточно зависимостей `.[dev]`. Полное сравнение использует `.[neural,dev]` для локального реранкера.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e ".[neural,dev]"
./scripts/download_data.sh
git -C data/mt-rag-benchmark rev-parse HEAD
python -m pytest -q
```

Скрипт загружает или обновляет копию IBM `mt-rag-benchmark` в `data/mt-rag-benchmark`. Для опубликованных здесь результатов использована ревизия `e0970de1b1207c2a9506adf238e6e1a22037e38a`. Если новая копия указывает на другую ревизию, перед индексацией загрузите и выберите использованный коммит:

```bash
git -C data/mt-rag-benchmark fetch --depth 1 origin e0970de1b1207c2a9506adf238e6e1a22037e38a
git -C data/mt-rag-benchmark switch --detach e0970de1b1207c2a9506adf238e6e1a22037e38a
```

Создайте `.env` из [шаблона](../.env.example), если файла ещё нет. Укажите полученные в лаборатории `LLM_BASE_URL` и `LLM_API_KEY`. В [config.yaml](../config.yaml) заданы имена моделей, параметры ретриверов и каталоги данных, индексов и прогонов. `arag index` строит dense-индексы по четырём общим корпусам. Они используются на обоих датасетах.

```bash
source .env
arag llm-check
arag rerank-check
arag index
```

## MTRAG

Запустите 30 фиксированных стратегий, постройте оракул и сравните роутеры. Подсчёт метрик использует сохранённые прогоны. Запуск `arag route` переиспользует их для выбранных комбинаций поиска.

```bash
arag run --dataset mtrag --strategies all
arag eval --dataset mtrag --slice later_turns
arag oracle --dataset mtrag
arag analyze --dataset mtrag

for strategy in last:bm25 last:dense last:hybrid+rerank llm_rewrite:dense+rerank; do
  arag route --dataset mtrag --router fixed --fixed "$strategy"
done
for router in rule llm_nohist llm_hist adaptive_rag oracle oracle_cheap; do
  arag route --dataset mtrag --router "$router"
done

arag routes-summary --dataset mtrag --slice later_turns
arag significance --dataset mtrag --slice later_turns --baseline 'llm_rewrite:dense+rerank'
arag significance --dataset mtrag --slice later_turns --baseline route/llm_nohist
```

Оцените девять систем и воспроизведите калибровку судьи на ответах с человеческими оценками.

```bash
arag judge-calibrate --out runs/judge_calibration_qwen_v2.md
arag answer --dataset mtrag --run \
  'runs/mtrag/last__dense.jsonl' \
  'runs/mtrag/last__hybrid+rerank.jsonl' \
  'runs/mtrag/llm_rewrite__dense+rerank.jsonl' \
  'runs/mtrag/chiq__dense+rerank.jsonl' \
  runs/mtrag_routes/{rule,adaptive_rag,llm_hist,llm_nohist,oracle}.jsonl

for slice in later_turns all; do
  arag answers-summary --dataset mtrag --slice "$slice" --baseline 'llm_rewrite:dense+rerank'
done
```

Ожидается 777 задач с qrels, включая 675 последующих ходов. [Поисковый отчёт](results/mtrag_analysis.md) · [Отчёт по ответам](results/mtrag_answers.md) · [Калибровка судьи](results/judge_calibration_qwen_v2.md)

## MTRAG-UN

Для переноса Adaptive-RAG обязательны полные прогоны трёх уровней на MTRAG. Если полный блок MTRAG выше пропущен, восстановите эти три прогона отдельно.

```bash
arag run --dataset mtrag --strategies last:dense last:dense+rerank llm_rewrite:dense+rerank
```

Основной эксперимент MTRAG-UN включает 12 стратегий, два режима обучения Adaptive-RAG и пару для проверки вклада истории. Команды с `--train-dataset` и без него пишут отдельные прогоны.

```bash
arag run --dataset mtragun --strategies \
  last:bm25 last:dense last:hybrid \
  last:bm25+rerank last:dense+rerank last:hybrid+rerank \
  llm_rewrite:bm25 llm_rewrite:dense llm_rewrite:hybrid \
  llm_rewrite:bm25+rerank llm_rewrite:dense+rerank llm_rewrite:hybrid+rerank

arag route --dataset mtragun --router fixed --fixed 'llm_rewrite:dense+rerank'
for router in rule llm_nohist llm_hist llm_nohist_rewrite adaptive_rag; do
  arag route --dataset mtragun --router "$router"
done
arag route --dataset mtragun --router adaptive_rag --train-dataset mtrag

arag answer --dataset mtragun --run \
  'runs/mtragun/llm_rewrite__dense+rerank.jsonl' \
  'runs/mtragun/llm_rewrite__hybrid+rerank.jsonl' \
  runs/mtragun_routes/{rule,llm_hist,llm_nohist,llm_nohist_rewrite,adaptive_rag,adaptive_rag_from_mtrag}.jsonl
```

### Сводки и парные тесты

Для основного сравнения используйте фиксированный dense-пайплайн. Для проверки вклада истории смотрите строку `route/llm_hist` в парных тестах относительно `route/llm_nohist_rewrite`. Для прямого сравнения режимов Adaptive-RAG смотрите строку `route/adaptive_rag_from_mtrag` относительно `route/adaptive_rag`.

```bash
arag eval --dataset mtragun --slice later_turns
for slice in later_turns all; do
  arag routes-summary --dataset mtragun --slice "$slice"
  arag answers-summary --dataset mtragun --slice "$slice" --baseline 'llm_rewrite:dense+rerank'
done

arag significance --dataset mtragun --slice later_turns --baseline 'llm_rewrite:dense+rerank'
arag significance --dataset mtragun --slice later_turns --baseline route/adaptive_rag
arag answers-summary --dataset mtragun --slice later_turns --baseline route/adaptive_rag
arag significance --dataset mtragun --slice later_turns --baseline route/llm_nohist_rewrite
arag answers-summary --dataset mtragun --slice later_turns --baseline route/llm_nohist_rewrite
arag answers-summary --dataset mtragun --slice later_turns --baseline route/llm_nohist_rewrite --metric completeness
```

Ожидается 332 задачи с qrels, включая 309 последующих ходов. Для переноса после исключения общих диалогов остаются 303 задачи, из них 283 последующих хода. Парные тесты используют общие ID сравниваемых систем. В сводке ответов `n` — количество ответов, `judged` — количество ответов с валидными оценками судьи. [Отчёт MTRAG-UN](results/mtragun.md)

## Файлы прогонов и повторный запуск

| Артефакт | Путь по умолчанию |
|---|---|
| Результаты стратегий | `runs/<dataset>/<strategy>.jsonl`. В имени файла двоеточие заменено на `__`. |
| Результаты роутеров | `runs/<dataset>_routes/<router>.jsonl`. |
| Ответы и оценки судьи | `runs/<dataset>_answers/*.jsonl`. |
| Сводки | `runs/<dataset>_summary.csv`, `runs/<dataset>_routes_summary_<slice>.csv`, `runs/<dataset>_answers_summary_<slice>.csv`. |
| Метки оракула и анализ поиска | `runs/<dataset>_oracle.jsonl`, `runs/<dataset>_analysis.md`. |
| Калибровка судьи | `runs/judge_calibration_<model>.md` и одноимённый JSONL. |

Команда `run` пропускает готовые файлы стратегий. Для пилотов с `--limit` или `--domains` задайте отдельный `runs_dir` в копии конфигурации и запускайте `arag --config <файл> <команда>`. Если усечённый файл уже записан, пересчитайте нужную стратегию с `--overwrite`. Параметр `answer --sample` выбирает одни и те же ID задач для всех переданных систем.

Парные тесты применяют поправку Холма к сравнениям в одной таблице. Для воспроизведения скорректированных p-значений используйте тот же набор файлов стратегий, роутеров и ответов, что в соответствующем эксперименте.

Ответы LLM, эмбеддинги и оценки пар реранкера переиспользуются из кэша. Для новых замеров задержки используйте отдельный кэш и `--workers 1` на этапах с LLM. В кэше остаётся время исходного вызова. При смене версии корпуса или параметров индексации используйте отдельный `cache_dir`, чтобы построить индексы заново.

Каталоги `runs/`, `cache/` и `data/` исключены из Git. Сохранённые отчёты перечислены в [индексе документации](README.md). Справка по CLI доступна через `arag <команда> --help`.
