"""Score every saved UltraFeedback classifier on the 300 paraphrased test responses, before vs after the rewrite.
  python -m src.paraphrase_eval   -> results/paraphrase_uf.json"""
import glob, json, os
import numpy as np, pandas as pd, torch, joblib
from . import text as T
from .explain import load
from .train import predict, loader

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "paraphrased_uf.jsonl"), encoding="utf-8")]
    df = pd.DataFrame(rows).drop_duplicates(["prompt_id", "family"]).sort_values(["family", "prompt_id"]).reset_index(drop=True)
    print(df.family.value_counts().to_dict())
    out = dict(n=len(df), paraphraser="Llama 3.2", before={}, after={}, after_pred_dist={}, labels=None)
    for path in sorted(glob.glob(os.path.join(HERE, "runs", "uf_*_output_s42.*"))):
        name = os.path.basename(path).split("_")[1]
        if "+" in name or name == "natural":   # tuned checkpoints need their cfg to rebuild; uf_natural models are a different dataset
            continue
        if path.endswith(".joblib"):
            ck = joblib.load(path); labels = ck["labels"]
            pb = ck["clf"].predict_proba(df.response.tolist()); pa = ck["clf"].predict_proba(df.paraphrase.tolist())
        else:
            model, vocab, ck = load(path); labels = ck["labels"]
            if ck["model"] == "charcnn":
                enc = lambda ts: [T.encode_chars(t) for t in ts]
            else:
                enc = lambda ts: [vocab.encode(T.tokenize(t)[:T.MAX_OUT]) for t in ts]
            pb = predict(model, loader(enc(df.response.tolist()), [0] * len(df), 128, False))
            pa = predict(model, loader(enc(df.paraphrase.tolist()), [0] * len(df), 128, False))
        y = df.family.map({l: i for i, l in enumerate(labels)}).values
        out["labels"] = labels
        out["before"][name] = float((pb.argmax(1) == y).mean())
        out["after"][name] = float((pa.argmax(1) == y).mean())
        out["after_pred_dist"][name] = {l: float((pa.argmax(1) == i).mean()) for i, l in enumerate(labels)}
        print(f"{name:8s} before={out['before'][name]:.3f} after={out['after'][name]:.3f}  predicted-after: " +
              ", ".join(f"{l}={v:.2f}" for l, v in out["after_pred_dist"][name].items()))
    json.dump(out, open(os.path.join(HERE, "results", "paraphrase_uf.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
