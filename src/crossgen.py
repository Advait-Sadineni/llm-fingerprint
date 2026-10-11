"""RQ3 cross-generation: run a classifier trained on the 2023 UltraFeedback families over the 2026 modern responses.
Reports, per modern family, the distribution of predicted 2023 families (does Llama-3 look like Llama-2?).
  python -m src.crossgen runs/uf_cnn_output_s42.pt"""
import json, os, sys
import numpy as np, pandas as pd, torch
from . import text as T
from .explain import load
from .train import predict, loader

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(path):
    model, vocab, ck = load(path)
    df = pd.read_csv(os.path.join(HERE, "data", "modern.csv.gz"), keep_default_na=False)
    seqs = [vocab.encode(T.view_tokens(i, o, ck["view"])) for i, o in zip(df.instruction, df.response)]
    probs = predict(model, loader(seqs, [0] * len(seqs), 128, False))
    pred = np.array(ck["labels"])[probs.argmax(1)]
    tab = pd.crosstab(df.family, pred, normalize="index").reindex(columns=ck["labels"], fill_value=0)
    out = dict(model=ck["model"], checkpoint=os.path.basename(path), labels=ck["labels"],
               uf_model_on_modern=tab.round(4).T.to_dict(), mean_confidence=float(probs.max(1).mean()))
    json.dump(out, open(os.path.join(HERE, "results", "crossgen.json"), "w"), indent=1)
    print(tab.round(3)); print("mean max-prob:", round(out["mean_confidence"], 3))


if __name__ == "__main__":
    main(sys.argv[1])
