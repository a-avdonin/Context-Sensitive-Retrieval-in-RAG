# Анализ: mtrag

Задач с qrels: 777. Метрика: `ndcg_cut_5`. Стратегии: `llm_rewrite:dense+rerank`, `llm_rewrite:hybrid+rerank`, `gold_rewrite:hybrid+rerank`, `gold_rewrite:dense+rerank`, `llm_rewrite:bm25+rerank`, `last:hybrid+rerank`, `llm_rewrite:dense`, `last:dense+rerank`, `gold_rewrite:bm25+rerank`, `gold_rewrite:dense`, `llm_rewrite:hybrid`, `last:bm25+rerank`, `last:dense`, `gold_rewrite:hybrid`, `last:hybrid`, `user_concat:hybrid+rerank`, `user_concat:dense+rerank`, `llm_rewrite:bm25`, `user_concat:bm25+rerank`, `user_concat:dense`, `user_concat:hybrid`, `gold_rewrite:bm25`, `last:bm25`, `user_concat:bm25`.

## 1. Состав датасета

### Домен

| Домен | задач | из них later | доля |
|---|---|---|---|
| clapnq | 208 | 180 | 0.268 |
| govt | 201 | 176 | 0.259 |
| cloud | 188 | 163 | 0.242 |
| fiqa | 180 | 156 | 0.232 |

### Номер хода пользователя

| Номер хода пользователя | задач | из них later | доля |
|---|---|---|---|
| 6+ | 271 | 271 | 0.349 |
| 2 | 106 | 106 | 0.136 |
| 3 | 103 | 103 | 0.133 |
| 1 | 102 | 0 | 0.131 |
| 4 | 99 | 99 | 0.127 |
| 5 | 96 | 96 | 0.124 |

### Multi-Turn

| Multi-Turn | задач | из них later | доля |
|---|---|---|---|
| Follow-up | 574 | 574 | 0.739 |
| N/A | 102 | 0 | 0.131 |
| Clarification | 101 | 101 | 0.130 |

### Answerability

| Answerability | задач | из них later | доля |
|---|---|---|---|
| ANSWERABLE | 709 | 620 | 0.912 |
| PARTIAL | 68 | 55 | 0.088 |

### Тип вопроса

| Тип вопроса | задач | из них later | доля |
|---|---|---|---|
| Factoid | 220 | 187 | 0.283 |
| Explanation | 134 | 119 | 0.172 |
| Summarization | 90 | 79 | 0.116 |
| How-To | 77 | 61 | 0.099 |
| Keyword | 57 | 55 | 0.073 |
| Opinion | 50 | 47 | 0.064 |
| Composite | 47 | 41 | 0.060 |
| Non-Question | 46 | 46 | 0.059 |
| Comparative | 44 | 31 | 0.057 |
| Troubleshooting | 12 | 9 | 0.015 |

Релевантных пассажей на запрос: среднее 2.74, распределение {1: 106, 2: 265, 3: 243, 4: 96, 5: 67} (5 = 5+).

## 2. Последний вопрос vs human rewrite (later turns)

Human rewrite есть у 675 из 675 later-задач; у остальных `gold_rewrite` = последний вопрос.

| показатель | значение |
|---|---|
| rewrite совпадает с вопросом | 0.121 |
| длина вопроса, слов | 8.316 |
| длина rewrite, слов | 9.776 |
| Jaccard контентных токенов | 0.510 |
| дефицит: доля токенов rewrite, которых нет в вопросе | 0.406 |
| из недостающих токенов есть в истории (все реплики) | 0.539 |
| из недостающих токенов есть в репликах пользователя | 0.332 |

## 3. Качество стратегий по срезам (`ndcg_cut_5`)

### По `turn`

| turn | n | llm_rewrite:dense+rerank | llm_rewrite:hybrid+rerank | gold_rewrite:hybrid+rerank | gold_rewrite:dense+rerank | llm_rewrite:bm25+rerank | last:hybrid+rerank | llm_rewrite:dense | last:dense+rerank | gold_rewrite:bm25+rerank | gold_rewrite:dense | llm_rewrite:hybrid | last:bm25+rerank | last:dense | gold_rewrite:hybrid | last:hybrid | user_concat:hybrid+rerank | user_concat:dense+rerank | llm_rewrite:bm25 | user_concat:bm25+rerank | user_concat:dense | user_concat:hybrid | gold_rewrite:bm25 | last:bm25 | user_concat:bm25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 102 | 0.620 | 0.623 | 0.623 | 0.620 | 0.562 | 0.623 | 0.499 | 0.620 | 0.562 | 0.499 | 0.473 | 0.562 | 0.499 | 0.473 | 0.473 | 0.623 | 0.620 | 0.335 | 0.562 | 0.499 | 0.473 | 0.335 | 0.335 | 0.335 |
| 2 | 106 | 0.507 | 0.513 | 0.445 | 0.445 | 0.483 | 0.386 | 0.433 | 0.377 | 0.416 | 0.386 | 0.382 | 0.360 | 0.367 | 0.348 | 0.320 | 0.402 | 0.402 | 0.314 | 0.390 | 0.375 | 0.363 | 0.278 | 0.203 | 0.277 |
| 3 | 103 | 0.427 | 0.425 | 0.404 | 0.401 | 0.349 | 0.336 | 0.377 | 0.340 | 0.360 | 0.363 | 0.316 | 0.298 | 0.308 | 0.307 | 0.272 | 0.192 | 0.207 | 0.234 | 0.175 | 0.208 | 0.201 | 0.222 | 0.185 | 0.156 |
| 4 | 99 | 0.405 | 0.379 | 0.378 | 0.377 | 0.349 | 0.376 | 0.348 | 0.382 | 0.333 | 0.338 | 0.332 | 0.317 | 0.300 | 0.299 | 0.273 | 0.181 | 0.176 | 0.245 | 0.180 | 0.215 | 0.188 | 0.195 | 0.158 | 0.144 |
| 5 | 96 | 0.403 | 0.411 | 0.373 | 0.362 | 0.353 | 0.368 | 0.384 | 0.370 | 0.305 | 0.342 | 0.311 | 0.317 | 0.308 | 0.252 | 0.269 | 0.198 | 0.177 | 0.243 | 0.181 | 0.164 | 0.186 | 0.194 | 0.207 | 0.141 |
| 6+ | 271 | 0.385 | 0.385 | 0.349 | 0.347 | 0.344 | 0.328 | 0.340 | 0.322 | 0.312 | 0.303 | 0.297 | 0.266 | 0.286 | 0.270 | 0.236 | 0.140 | 0.141 | 0.199 | 0.131 | 0.137 | 0.154 | 0.167 | 0.157 | 0.112 |

### По `domain`

| domain | n | llm_rewrite:dense+rerank | llm_rewrite:hybrid+rerank | gold_rewrite:hybrid+rerank | gold_rewrite:dense+rerank | llm_rewrite:bm25+rerank | last:hybrid+rerank | llm_rewrite:dense | last:dense+rerank | gold_rewrite:bm25+rerank | gold_rewrite:dense | llm_rewrite:hybrid | last:bm25+rerank | last:dense | gold_rewrite:hybrid | last:hybrid | user_concat:hybrid+rerank | user_concat:dense+rerank | llm_rewrite:bm25 | user_concat:bm25+rerank | user_concat:dense | user_concat:hybrid | gold_rewrite:bm25 | last:bm25 | user_concat:bm25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clapnq | 208 | 0.520 | 0.500 | 0.477 | 0.476 | 0.447 | 0.432 | 0.480 | 0.437 | 0.426 | 0.466 | 0.392 | 0.385 | 0.398 | 0.378 | 0.340 | 0.289 | 0.289 | 0.261 | 0.293 | 0.291 | 0.274 | 0.250 | 0.206 | 0.188 |
| cloud | 188 | 0.417 | 0.433 | 0.370 | 0.359 | 0.368 | 0.411 | 0.309 | 0.393 | 0.329 | 0.275 | 0.310 | 0.335 | 0.296 | 0.261 | 0.274 | 0.216 | 0.208 | 0.252 | 0.202 | 0.176 | 0.214 | 0.188 | 0.194 | 0.160 |
| fiqa | 180 | 0.345 | 0.339 | 0.344 | 0.347 | 0.296 | 0.286 | 0.316 | 0.292 | 0.289 | 0.290 | 0.259 | 0.233 | 0.282 | 0.236 | 0.200 | 0.210 | 0.217 | 0.156 | 0.177 | 0.192 | 0.166 | 0.144 | 0.108 | 0.102 |
| govt | 201 | 0.475 | 0.481 | 0.445 | 0.442 | 0.450 | 0.408 | 0.419 | 0.405 | 0.412 | 0.382 | 0.387 | 0.373 | 0.343 | 0.365 | 0.343 | 0.309 | 0.306 | 0.315 | 0.281 | 0.289 | 0.292 | 0.280 | 0.269 | 0.250 |

### По `multi_turn`

| multi_turn | n | llm_rewrite:dense+rerank | llm_rewrite:hybrid+rerank | gold_rewrite:hybrid+rerank | gold_rewrite:dense+rerank | llm_rewrite:bm25+rerank | last:hybrid+rerank | llm_rewrite:dense | last:dense+rerank | gold_rewrite:bm25+rerank | gold_rewrite:dense | llm_rewrite:hybrid | last:bm25+rerank | last:dense | gold_rewrite:hybrid | last:hybrid | user_concat:hybrid+rerank | user_concat:dense+rerank | llm_rewrite:bm25 | user_concat:bm25+rerank | user_concat:dense | user_concat:hybrid | gold_rewrite:bm25 | last:bm25 | user_concat:bm25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Clarification | 101 | 0.442 | 0.425 | 0.382 | 0.368 | 0.382 | 0.269 | 0.362 | 0.275 | 0.340 | 0.310 | 0.344 | 0.245 | 0.230 | 0.277 | 0.199 | 0.220 | 0.227 | 0.260 | 0.218 | 0.206 | 0.216 | 0.216 | 0.142 | 0.168 |
| Follow-up | 574 | 0.411 | 0.412 | 0.380 | 0.379 | 0.366 | 0.366 | 0.369 | 0.362 | 0.337 | 0.341 | 0.316 | 0.310 | 0.320 | 0.292 | 0.277 | 0.200 | 0.198 | 0.231 | 0.188 | 0.200 | 0.201 | 0.198 | 0.182 | 0.151 |
| N/A | 102 | 0.620 | 0.623 | 0.623 | 0.620 | 0.562 | 0.623 | 0.499 | 0.620 | 0.562 | 0.499 | 0.473 | 0.562 | 0.499 | 0.473 | 0.473 | 0.623 | 0.620 | 0.335 | 0.562 | 0.499 | 0.473 | 0.335 | 0.335 | 0.335 |

### По `answerability`

| answerability | n | llm_rewrite:dense+rerank | llm_rewrite:hybrid+rerank | gold_rewrite:hybrid+rerank | gold_rewrite:dense+rerank | llm_rewrite:bm25+rerank | last:hybrid+rerank | llm_rewrite:dense | last:dense+rerank | gold_rewrite:bm25+rerank | gold_rewrite:dense | llm_rewrite:hybrid | last:bm25+rerank | last:dense | gold_rewrite:hybrid | last:hybrid | user_concat:hybrid+rerank | user_concat:dense+rerank | llm_rewrite:bm25 | user_concat:bm25+rerank | user_concat:dense | user_concat:hybrid | gold_rewrite:bm25 | last:bm25 | user_concat:bm25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ANSWERABLE | 709 | 0.447 | 0.448 | 0.419 | 0.415 | 0.399 | 0.400 | 0.391 | 0.396 | 0.370 | 0.366 | 0.340 | 0.343 | 0.345 | 0.315 | 0.299 | 0.260 | 0.259 | 0.246 | 0.245 | 0.238 | 0.238 | 0.216 | 0.199 | 0.178 |
| PARTIAL | 68 | 0.393 | 0.375 | 0.345 | 0.342 | 0.338 | 0.249 | 0.323 | 0.260 | 0.335 | 0.268 | 0.344 | 0.243 | 0.200 | 0.294 | 0.217 | 0.241 | 0.241 | 0.270 | 0.207 | 0.253 | 0.245 | 0.239 | 0.173 | 0.168 |

### По `question_type`

| question_type | n | llm_rewrite:dense+rerank | llm_rewrite:hybrid+rerank | gold_rewrite:hybrid+rerank | gold_rewrite:dense+rerank | llm_rewrite:bm25+rerank | last:hybrid+rerank | llm_rewrite:dense | last:dense+rerank | gold_rewrite:bm25+rerank | gold_rewrite:dense | llm_rewrite:hybrid | last:bm25+rerank | last:dense | gold_rewrite:hybrid | last:hybrid | user_concat:hybrid+rerank | user_concat:dense+rerank | llm_rewrite:bm25 | user_concat:bm25+rerank | user_concat:dense | user_concat:hybrid | gold_rewrite:bm25 | last:bm25 | user_concat:bm25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Comparative | 44 | 0.484 | 0.499 | 0.477 | 0.451 | 0.427 | 0.436 | 0.433 | 0.446 | 0.411 | 0.416 | 0.389 | 0.366 | 0.361 | 0.336 | 0.303 | 0.389 | 0.372 | 0.287 | 0.364 | 0.285 | 0.326 | 0.251 | 0.199 | 0.252 |
| Composite | 47 | 0.576 | 0.578 | 0.525 | 0.524 | 0.469 | 0.499 | 0.522 | 0.495 | 0.398 | 0.473 | 0.393 | 0.378 | 0.468 | 0.370 | 0.348 | 0.341 | 0.345 | 0.277 | 0.328 | 0.355 | 0.345 | 0.223 | 0.242 | 0.240 |
| Explanation | 134 | 0.409 | 0.428 | 0.401 | 0.387 | 0.387 | 0.387 | 0.379 | 0.364 | 0.356 | 0.342 | 0.298 | 0.336 | 0.310 | 0.308 | 0.267 | 0.226 | 0.217 | 0.200 | 0.230 | 0.210 | 0.185 | 0.188 | 0.167 | 0.146 |
| Factoid | 220 | 0.471 | 0.461 | 0.422 | 0.420 | 0.414 | 0.381 | 0.411 | 0.368 | 0.396 | 0.377 | 0.350 | 0.343 | 0.338 | 0.311 | 0.289 | 0.272 | 0.270 | 0.256 | 0.251 | 0.269 | 0.253 | 0.222 | 0.197 | 0.173 |
| How-To | 77 | 0.397 | 0.403 | 0.360 | 0.373 | 0.363 | 0.325 | 0.359 | 0.341 | 0.331 | 0.324 | 0.315 | 0.284 | 0.328 | 0.276 | 0.266 | 0.273 | 0.268 | 0.199 | 0.253 | 0.251 | 0.236 | 0.173 | 0.144 | 0.160 |
| Keyword | 57 | 0.296 | 0.269 | 0.288 | 0.284 | 0.241 | 0.360 | 0.220 | 0.380 | 0.251 | 0.253 | 0.241 | 0.312 | 0.286 | 0.241 | 0.309 | 0.180 | 0.192 | 0.250 | 0.169 | 0.148 | 0.198 | 0.205 | 0.260 | 0.172 |
| Non-Question | 46 | 0.516 | 0.492 | 0.436 | 0.437 | 0.456 | 0.393 | 0.414 | 0.421 | 0.381 | 0.307 | 0.429 | 0.370 | 0.265 | 0.337 | 0.308 | 0.176 | 0.180 | 0.298 | 0.167 | 0.185 | 0.200 | 0.254 | 0.237 | 0.182 |
| Opinion | 50 | 0.455 | 0.464 | 0.454 | 0.449 | 0.420 | 0.367 | 0.415 | 0.379 | 0.373 | 0.406 | 0.381 | 0.301 | 0.368 | 0.357 | 0.308 | 0.256 | 0.251 | 0.299 | 0.181 | 0.204 | 0.198 | 0.272 | 0.211 | 0.142 |
| Summarization | 90 | 0.404 | 0.398 | 0.370 | 0.374 | 0.359 | 0.367 | 0.332 | 0.370 | 0.341 | 0.327 | 0.332 | 0.309 | 0.304 | 0.314 | 0.275 | 0.234 | 0.243 | 0.224 | 0.233 | 0.216 | 0.248 | 0.200 | 0.154 | 0.184 |
| Troubleshooting | 12 | 0.578 | 0.594 | 0.654 | 0.654 | 0.530 | 0.607 | 0.390 | 0.552 | 0.570 | 0.429 | 0.433 | 0.518 | 0.405 | 0.443 | 0.467 | 0.355 | 0.375 | 0.472 | 0.272 | 0.322 | 0.306 | 0.486 | 0.482 | 0.310 |

### По `dependent` (`dependent` — эвристика RuleRouter: later-ход с местоимением, короткий или без вопроса; first_turn — первый ход)

| dependent | n | llm_rewrite:dense+rerank | llm_rewrite:hybrid+rerank | gold_rewrite:hybrid+rerank | gold_rewrite:dense+rerank | llm_rewrite:bm25+rerank | last:hybrid+rerank | llm_rewrite:dense | last:dense+rerank | gold_rewrite:bm25+rerank | gold_rewrite:dense | llm_rewrite:hybrid | last:bm25+rerank | last:dense | gold_rewrite:hybrid | last:hybrid | user_concat:hybrid+rerank | user_concat:dense+rerank | llm_rewrite:bm25 | user_concat:bm25+rerank | user_concat:dense | user_concat:hybrid | gold_rewrite:bm25 | last:bm25 | user_concat:bm25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| False | 290 | 0.459 | 0.458 | 0.409 | 0.412 | 0.397 | 0.407 | 0.421 | 0.406 | 0.362 | 0.381 | 0.337 | 0.342 | 0.377 | 0.302 | 0.296 | 0.216 | 0.215 | 0.224 | 0.207 | 0.222 | 0.206 | 0.191 | 0.179 | 0.147 |
| True | 385 | 0.384 | 0.381 | 0.359 | 0.351 | 0.347 | 0.309 | 0.328 | 0.306 | 0.320 | 0.302 | 0.308 | 0.269 | 0.254 | 0.280 | 0.242 | 0.193 | 0.193 | 0.244 | 0.182 | 0.185 | 0.201 | 0.208 | 0.173 | 0.159 |
| first_turn | 102 | 0.620 | 0.623 | 0.623 | 0.620 | 0.562 | 0.623 | 0.499 | 0.620 | 0.562 | 0.499 | 0.473 | 0.562 | 0.499 | 0.473 | 0.473 | 0.623 | 0.620 | 0.335 | 0.562 | 0.499 | 0.473 | 0.335 | 0.335 | 0.335 |

## 4. Оракул

Unsolvable — доля запросов, где ни одна стратегия не дала ненулевой метрики.

| срез | n | unsolvable (deployable) | unsolvable (+gold) | oracle deployable | oracle +gold |
|---|---|---|---|---|---|
| domain=clapnq | 208 | 0.077 | 0.072 | 0.665 | 0.680 |
| domain=cloud | 188 | 0.101 | 0.085 | 0.630 | 0.642 |
| domain=fiqa | 180 | 0.161 | 0.128 | 0.542 | 0.570 |
| domain=govt | 201 | 0.119 | 0.095 | 0.676 | 0.698 |
| turn=1 | 102 | 0.059 | 0.059 | 0.707 | 0.707 |
| turn=2 | 106 | 0.047 | 0.038 | 0.701 | 0.716 |
| turn=3 | 103 | 0.126 | 0.097 | 0.621 | 0.640 |
| turn=4 | 99 | 0.101 | 0.071 | 0.628 | 0.661 |
| turn=5 | 96 | 0.115 | 0.094 | 0.625 | 0.643 |
| turn=6+ | 271 | 0.159 | 0.137 | 0.582 | 0.605 |

## 5. Когда human rewrite помогает BM25 (later turns с rewrite)

win/tie/loss — сравнение `gold_rewrite:bm25` с `last:bm25` по запросу.

| срез | n | Δ mean | win | tie | loss |
|---|---|---|---|---|---|
| multi_turn=Clarification | 101 | 0.075 | 23 | 70 | 8 |
| multi_turn=Follow-up | 574 | 0.016 | 100 | 394 | 80 |
| dependent=False | 290 | 0.012 | 44 | 208 | 38 |
| dependent=True | 385 | 0.035 | 79 | 256 | 50 |
| question_type=Comparative | 31 | 0.074 | 7 | 20 | 4 |
| question_type=Composite | 41 | -0.022 | 6 | 28 | 7 |
| question_type=Explanation | 119 | 0.024 | 23 | 79 | 17 |
| question_type=Factoid | 187 | 0.029 | 34 | 136 | 17 |
| question_type=How-To | 61 | 0.038 | 10 | 43 | 8 |
| question_type=Keyword | 55 | -0.057 | 7 | 36 | 12 |
| question_type=Non-Question | 46 | 0.017 | 5 | 36 | 5 |
| question_type=Opinion | 47 | 0.065 | 11 | 30 | 6 |
| question_type=Summarization | 79 | 0.052 | 17 | 52 | 10 |
| question_type=Troubleshooting | 9 | 0.004 | 3 | 4 | 2 |

### Выигрыш от rewrite в зависимости от дефицита контекста

| дефицит | n | last | gold | Δ mean | win | loss |
|---|---|---|---|---|---|---|
| 0 | 177 | 0.240 | 0.234 | -0.005 | 10 | 13 |
| (0, 1/3) | 134 | 0.220 | 0.226 | 0.006 | 26 | 23 |
| [1/3, 2/3) | 215 | 0.161 | 0.187 | 0.026 | 49 | 36 |
| [2/3, 1] | 149 | 0.081 | 0.158 | 0.077 | 38 | 16 |
