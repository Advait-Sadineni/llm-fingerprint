"""Collect results/*.json into results/summary.csv (mean +- std over seeds) and results/stats.json
(McNemar CNN vs LSTM, calibration ECE). `python -m src.aggregate` prints the headline tables."""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
from .train import metrics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ["data", "model", "view", "train_src", "test_src", "ablate", "frac"]
ENSEMBLE = ["tfidf", "cnn", "lstm", "bert"]


def load_runs():
    runs = []
    for f in glob.glob(os.path.join(HERE, "results", "*.json")):
        if os.path.basename(f).split("_")[0] in ("features", "explain", "stats", "paraphrase", "crossgen"):
            continue
        r = json.load(open(f))
        if "args" not in r or r["args"].get("smoke"):
            continue
        if "test_probs" not in r:                                  # predictions live in a compressed sidecar to keep the repo small
            p = os.path.join(HERE, "results", "preds", os.path.basename(f)[:-5] + ".npz")
            if os.path.exists(p):
                z = np.load(p, allow_pickle=False)
                r["test_probs"], r["test_y"], r["test_prompt_ids"] = z["probs"].tolist(), z["y"].tolist(), z["prompt_ids"].tolist()
        runs.append(r)
    return runs


def row(r):
    a = r["args"]
    d = {k: a.get(k) for k in KEYS}
    d["model"] = a["model"] + ("+" + a["tag"] if a.get("tag") else "")
    return dict(d, cfg=a.get("cfg", ""), seed=a["seed"], accuracy=r["accuracy"], macro_f1=r["macro_f1"],
                n_test=r["n_test"], secs=r["secs"], epochs=len(r["history"]),
                val_f1=max((h["val_macro_f1"] for h in r["history"]), default=float("nan")))


def summary(runs):
    df = pd.DataFrame([row(r) for r in runs]).fillna({"train_src": "", "test_src": "", "ablate": "none", "frac": 1.0, "cfg": ""})
    sweep = df[(df.cfg != "") & ~df.model.str.contains(r"\+")]           # untagged cfg runs = hyper-parameter sweep (val only)
    sweep[["data", "model", "view", "cfg", "seed", "val_f1", "accuracy", "macro_f1"]].sort_values(["model", "val_f1"], ascending=[True, False]).to_csv(
        os.path.join(HERE, "results", "sweep.csv"), index=False)
    df = df[~df.index.isin(sweep.index)].drop(columns=["cfg", "val_f1"])
    g = df.groupby(KEYS)
    out = g.agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                seeds=("seed", "count"), n_test=("n_test", "first"), secs=("secs", "mean"), epochs=("epochs", "mean")).reset_index()
    return out.fillna({"acc_std": 0, "f1_std": 0})


def mcnemar(a, b):
    """Exact McNemar on paired correctness vectors."""
    ya, yb = np.array(a["test_y"]), np.array(b["test_y"])
    assert (ya == yb).all() and a["test_prompt_ids"] == b["test_prompt_ids"]
    ca, cb = np.array(a["test_probs"]).argmax(1) == ya, np.array(b["test_probs"]).argmax(1) == yb
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(a_only_correct=n01, b_only_correct=n10, p_value=float(p))


def ece(r, bins=10):
    p = np.array(r["test_probs"]); y = np.array(r["test_y"])
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def source_oracle(data):
    """Accuracy of predicting the family from the prompt SOURCE alone (majority family per source in train).
    This is the ceiling an input-only classifier can reach by recognising the source rather than the model."""
    df = pd.read_csv(os.path.join(HERE, "data", data + ".csv.gz"), keep_default_na=False)
    maj = df[df.split == "train"].groupby("source").family.agg(lambda s: s.value_counts().idxmax())
    te = df[df.split == "test"]
    return float((te.source.map(maj) == te.family).mean())


def ensemble(by_name, data, seed):
    names = [f"{data}_{m}_output_s{42 if m == 'tfidf' else seed}" for m in ENSEMBLE]
    rs = [by_name[n] for n in names if n in by_name]
    if len(rs) < 3:
        return None
    y = np.array(rs[0]["test_y"]); assert all(r["test_prompt_ids"] == rs[0]["test_prompt_ids"] for r in rs)
    p = np.mean([np.array(r["test_probs"]) for r in rs], axis=0)
    fake = dict(name=f"{data}_ensemble_output_s{seed}", args=dict(data=data, model="ensemble", view="output", seed=seed), labels=rs[0]["labels"],
                n_test=len(y), secs=0, history=[], test_prompt_ids=rs[0]["test_prompt_ids"], test_y=rs[0]["test_y"], test_probs=p.tolist(),
                members=[r["name"] for r in rs], **metrics(y, p, rs[0]["labels"]))
    return fake


def main():
    runs = load_runs()
    by_name = {r["name"]: r for r in runs}
    for data in ["uf", "modern"]:
        for seed in [42, 1, 2]:
            e = ensemble(by_name, data, seed)
            if e:
                runs.append(e); by_name[e["name"]] = e
    df = summary(runs)
    df.to_csv(os.path.join(HERE, "results", "summary.csv"), index=False)
    stats = {}
    for data in ["uf", "modern", "uf17"]:
        pair = (f"{data}_cnn_output_s42", f"{data}_lstm_output_s42")
        if all(n in by_name for n in pair):
            stats[f"mcnemar_cnn_vs_lstm_{data}"] = mcnemar(by_name[pair[0]], by_name[pair[1]])
        e = f"{data}_ensemble_output_s42"
        if e in by_name:
            stats[f"ensemble_members_{data}"] = by_name[e]["members"]
            best = max((m for m in ENSEMBLE if f"{data}_{m}_output_s42" in by_name), key=lambda m: by_name[f"{data}_{m}_output_s42"]["macro_f1"])
            stats[f"mcnemar_ensemble_vs_{best}_{data}"] = mcnemar(by_name[e], by_name[f"{data}_{best}_output_s42"])
        for m in ["tfidf", "cnn", "lstm", "cnnlstm", "charcnn", "bert"]:
            n = f"{data}_{m}_output_s42"
            if n in by_name:
                stats[f"ece_{n}"] = ece(by_name[n])
    for data in ["uf", "uf_natural"]:
        stats[f"source_oracle_{data}"] = source_oracle(data)
    json.dump(stats, open(os.path.join(HERE, "results", "stats.json"), "w"), indent=1)
    pd.set_option("display.width", 200)
    base = df[(df.train_src == "") & (df.ablate == "none") & (df.frac == 1.0)]
    for data in sorted(base.data.unique()):
        print(f"\n== {data}")
        t = base[base.data == data].pivot_table(index="model", columns="view", values="f1_mean").round(3)
        s = base[base.data == data].pivot_table(index="model", columns="view", values="seeds")
        print("macro-F1 (seeds):"); print(t.astype(str) + " (" + s.fillna(0).astype(int).astype(str) + ")")
    print("\nstats:", json.dumps(stats, indent=1)[:1500])


if __name__ == "__main__":
    main()
