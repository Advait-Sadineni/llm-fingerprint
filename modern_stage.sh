#!/usr/bin/env bash
# Everything that depends on the modern set, in order. Run after data/generate_modern.py has finished.
set -e
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
python data/build_modern.py
bash experiments.sh modern
bash experiments.sh loso
for s in 42 1 2; do
  python run.py --data modern --model cnn --view output --seed $s --cfg emb=300,channels=128,kernels=2-3-4-5 --tag tuned
  python run.py --data modern --model lstm --view output --seed $s --cfg emb=300,hidden=256,layers=1 --tag tuned
done
python -m src.features modern
python -m src.explain runs/modern_cnn_output_s42.pt
python -m src.explain runs/modern_lstm_output_s42.pt
python -m src.crossgen runs/uf_cnn_output_s42.pt
python -m src.aggregate
python -m src.tables
python -m src.plots
echo "modern_stage.sh done"
