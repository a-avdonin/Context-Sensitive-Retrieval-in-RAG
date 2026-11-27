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

Команды ниже рассчитаны на Bash или Zsh и отдельный пустой каталог прогонов при воспроизведении с нуля. Сначала запустите 18 стратегий и сохраните выбор оракула. Затем расширьте поиск до 30 стратегий, включая CHIQ и человеческие формулировки. Такой порядок воспроизводит строку оракула в README.

```bash
arag run --dataset mtrag --strategies {last,user_concat,llm_rewrite}:{bm25,dense,hybrid}{,+rerank}
arag oracle --dataset mtrag
for router in oracle oracle_cheap; do
  arag route --dataset mtrag --router "$router"
done

arag run --dataset mtrag --strategies all
arag analyze --dataset mtrag
for strategy in last:bm25 last:dense last:hybrid+rerank llm_rewrite:dense+rerank; do
  arag route --dataset mtrag --router fixed --fixed "$strategy"
done
for router in rule llm_nohist llm_hist adaptive_rag; do
  arag route --dataset mtrag --router "$router"
done

arag answer --dataset mtrag --k 5 --run \
  runs/mtrag/{last__bm25,last__dense,last__dense+rerank,last__hybrid+rerank}.jsonl \
  runs/mtrag/{llm_rewrite__dense+rerank,llm_rewrite__hybrid+rerank,chiq__dense+rerank,gold_rewrite__dense+rerank}.jsonl \
  runs/mtrag_routes/{rule,llm_nohist,llm_hist,adaptive_rag,oracle}.jsonl
```

Ожидаются ответы 13 систем по 777 задачам с qrels, включая 675 последующих ходов. Отдельная команда `arag judge-calibrate --out runs/judge_calibration_qwen_v2.md` проверяет судью на ответах с человеческими оценками. [Калибровка v2](results/judge_calibration_qwen_v2.md)

## MTRAG-UN

Сохраните решения оракула по 12 исходным конфигурациям до запуска CHIQ. Для переноса нужны сохранённые прогоны трёх уровней Adaptive-RAG из блока MTRAG.

```bash
arag run --dataset mtragun --strategies {last,llm_rewrite}:{bm25,dense,hybrid}{,+rerank}
arag oracle --dataset mtragun
arag route --dataset mtragun --router oracle
arag run --dataset mtragun --strategies chiq:dense+rerank

arag route --dataset mtragun --router fixed --fixed llm_rewrite:dense+rerank
for router in rule llm_nohist llm_hist llm_nohist_rewrite adaptive_rag; do
  arag route --dataset mtragun --router "$router"
done
arag route --dataset mtragun --router adaptive_rag --train-dataset mtrag

arag answer --dataset mtragun --k 5 --run \
  runs/mtragun/{last__bm25,last__dense,last__dense+rerank,last__hybrid+rerank}.jsonl \
  runs/mtragun/{llm_rewrite__dense+rerank,llm_rewrite__hybrid+rerank,chiq__dense+rerank}.jsonl \
  runs/mtragun_routes/{rule,llm_nohist,llm_nohist_rewrite,llm_hist,adaptive_rag,adaptive_rag_from_mtrag,oracle}.jsonl
```

Сводка включает 14 систем. Для переноса ожидаются 303 ответа, включая 283 последующих хода. Для остальных систем — 332 ответа, включая 309 последующих ходов.

## Сводки и парные тесты

Команды читают сохранённые прогоны. Для сравнения по Correctness применяется `answers-summary`, для nDCG@5 — `significance`.

```bash
for ds in mtrag mtragun; do
  arag eval --dataset "$ds" --slice later_turns
  for slice in all later_turns; do
    arag routes-summary --dataset "$ds" --slice "$slice"
    arag answers-summary --dataset "$ds" --slice "$slice" --baseline llm_rewrite:dense+rerank
  done
  arag significance --dataset "$ds" --slice later_turns --baseline llm_rewrite:dense+rerank
done
```

Для контроля истории на MTRAG-UN замените базу на `route/llm_nohist_rewrite` и прочитайте строку `route/llm_hist`. Для переноса сравните `route/adaptive_rag_from_mtrag` с базой `route/adaptive_rag`. Для Completeness добавьте к `answers-summary` аргумент `--metric completeness`. [Интерпретация парных тестов](methodology.md#значимость)

## Файлы прогонов и повторный запуск

Поиск сохраняется в `runs/<dataset>/`, выбор роутера — в `runs/<dataset>_routes/`, ответы и оценки судьи — в `runs/<dataset>_answers/`. В имени файла стратегии двоеточие заменяется на `__`. CSV-сводки находятся непосредственно в `runs/`. Поля `n` и `judged` показывают число ответов и число валидных оценок. В Git хранятся отчёты, а данные, индексы, кэш и JSONL-прогоны остаются локально.

Для обновления таблиц по готовым JSONL достаточно блока со сводками. Если имеются только поисковые прогоны, начните с соответствующей команды `arag answer`. Готовый выбор оракула используйте из `<dataset>_routes/oracle.jsonl`: команда `arag oracle` при повторном запуске строит метки по всем стратегиям, доступным в этот момент.

Для пилота задайте отдельный `runs_dir` в копии конфигурации и используйте `arag --config <файл> <команда>`: команда `run` пропускает существующие файлы. Для новых замеров задержки используйте отдельный `cache_dir` и `--workers 1` на этапах с LLM. При смене корпуса или параметров индексации перестройте индексы в новом каталоге кэша.
