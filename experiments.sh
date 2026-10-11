#!/usr/bin/env bash
# Every run behind the report, in order. Each run is skipped if its results JSON already exists.
# Usage: bash experiments.sh [stage]   stages: core modern fine loso curves ablate all
set -e
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
stage=${1:-all}
SEEDS="42 1 2"
ALL="evol_instruct,false_qa,flan,sharegpt,truthful_qa,ultrachat"

if [[ $stage == core || $stage == all ]]; then           # RQ1 + RQ2 on the controlled UltraFeedback set
  for s in $SEEDS; do for v in output input both; do for m in tfidf cnn lstm cnnlstm charcnn; do
    python run.py --data uf --model $m --view $v --seed $s
  done; done; done
  for v in output input both; do python run.py --data uf --model bert --view $v --seed 42; done
  for v in output input both; do for m in tfidf cnn lstm; do python run.py --data uf_natural --model $m --view $v --seed 42; done; done
fi
if [[ $stage == fine || $stage == all ]]; then           # 16-way fine-grained model id
  for m in tfidf cnn lstm bert; do python run.py --data uf17 --model $m --view output --seed 42; done
fi
if [[ $stage == modern || $stage == all ]]; then         # RQ1 + RQ2 on the self-built modern set
  for s in $SEEDS; do for v in output input both; do for m in tfidf cnn lstm cnnlstm charcnn; do
    python run.py --data modern --model $m --view $v --seed $s
  done; done; done
  for v in output both; do python run.py --data modern --model bert --view $v --seed 42; done
fi
if [[ $stage == loso || $stage == all ]]; then           # RQ3 leave-one-source-out
  for d in uf modern; do [ -f data/$d.csv.gz ] || continue; for held in evol_instruct false_qa flan sharegpt truthful_qa ultrachat; do
    rest=$(echo $ALL | tr ',' '\n' | grep -v "^$held$" | paste -sd,)
    for m in tfidf cnn lstm; do python run.py --data $d --model $m --view output --train-src $rest --test-src $held --seed 42; done
  done; done
fi
if [[ $stage == curves || $stage == all ]]; then         # RQ3 learning curves
  for f in 0.05 0.1 0.25 0.5; do for m in cnn lstm; do for s in $SEEDS; do
    python run.py --data uf --model $m --view output --frac $f --seed $s
  done; done; done
fi
if [[ $stage == ablate || $stage == all ]]; then         # RQ4 ablations
  for a in strip_format cap100 first50 function_words shuffle; do for m in cnn lstm; do for s in $SEEDS; do
    python run.py --data uf --model $m --view output --ablate $a --seed $s
  done; done; done
fi
echo "experiments.sh $stage done"
