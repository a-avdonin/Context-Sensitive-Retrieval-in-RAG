#!/usr/bin/env bash
# MTRAG + MTRAG-UN (IBM/mt-rag-benchmark): корпуса, задачи, qrels
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -d data/mt-rag-benchmark ]; then
  git clone --depth 1 https://github.com/IBM/mt-rag-benchmark.git data/mt-rag-benchmark
else
  git -C data/mt-rag-benchmark pull --ff-only
fi
ls -lh data/mt-rag-benchmark/corpora/passage_level
