#!/usr/bin/env bash
# Hyper-parameter selection for the two required models, on the VALIDATION set only (seed 42, uf, output view).
# Selection rule (fixed before running): best val macro-F1; adopt as "tuned" only if >= 0.01 better than the default on val.
set -e
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
for emb in 128 300; do for ch in 128 256; do for k in 3-4-5 2-3-4-5; do
  python run.py --data uf --model cnn --view output --seed 42 --cfg emb=$emb,channels=$ch,kernels=$k
done; done; done
for emb in 128 300; do for h in 128 256; do for l in 1 2; do
  python run.py --data uf --model lstm --view output --seed 42 --cfg emb=$emb,hidden=$h,layers=$l
done; done; done
echo "sweep.sh done"
