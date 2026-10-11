"""RQ4: hand-crafted stylometric features. `python -m src.features uf` writes results/features_<data>.json
with per-family means, a feature-importance ranking, and the accuracy of a classifier that sees ONLY these features."""
import json, os, re, sys
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import f_classif
from .text import tokenize

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEDGES = ["I'm sorry", "I am sorry", "As an AI", "as an AI", "I cannot", "I can't", "I apologize", "Unfortunately", "However", "It's important to note", "Please note"]
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def featurize(text):
    toks = tokenize(text)
    words = [t for t in toks if t[0].isalnum()]
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    lines = text.split("\n")
    first200 = words[:200]
    return dict(
        chars=len(text), words=len(words), sentences=len(sents),
        avg_sent_len=len(words) / max(1, len(sents)),
        avg_word_len=np.mean([len(w) for w in words]) if words else 0,
        ttr200=len(set(w.lower() for w in first200)) / max(1, len(first200)),
        newlines=text.count("\n"), blank_lines=sum(1 for l in lines if not l.strip()),
        bullets=sum(1 for l in lines if re.match(r"\s*[-*•]\s", l)),
        numbered=sum(1 for l in lines if re.match(r"\s*\d+[.)]\s", l)),
        headers=sum(1 for l in lines if l.lstrip().startswith("#")),
        bold=text.count("**") // 2, code=text.count("```") // 2, inline_code=text.count("`") - 6 * (text.count("```") // 2),
        exclaim=text.count("!"), question=text.count("?"), colon=text.count(":"), semicolon=text.count(";"),
        commas_per_sent=text.count(",") / max(1, len(sents)),
        hedges=sum(text.count(h) for h in HEDGES), emoji=len(EMOJI_RE.findall(text)),
        caps_ratio=sum(1 for w in words if w.isupper() and len(w) > 1) / max(1, len(words)),
        digits_ratio=sum(1 for t in toks if t.isdigit()) / max(1, len(toks)),
        starts_with_sure=int(bool(re.match(r"\s*(Sure|Certainly|Of course|Absolutely)", text))),
        ends_with_question=int(text.rstrip().endswith("?")),
    )


def main(data="uf"):
    df = pd.read_csv(os.path.join(HERE, "data", data + ".csv.gz"), keep_default_na=False)
    F = pd.DataFrame([featurize(t) for t in df.response])
    y = df.family.values
    tr, te = (df.split == "train").values, (df.split == "test").values
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=0).fit(F[tr], y[tr])
    pred = clf.predict(F[te])
    fstat, _ = f_classif(F[tr], y[tr])
    out = dict(data=data, features=list(F.columns),
               per_family_mean=F.groupby(y).mean().round(3).to_dict(orient="index"),
               per_family_median=F.groupby(y).median().round(3).to_dict(orient="index"),
               anova_f=dict(sorted(zip(F.columns, np.nan_to_num(fstat).round(1).tolist()), key=lambda kv: -kv[1])),
               gbm_importance=dict(sorted(zip(F.columns, clf.feature_importances_.round(4).tolist()), key=lambda kv: -kv[1])),
               features_only_accuracy=float(accuracy_score(y[te], pred)),
               features_only_macro_f1=float(f1_score(y[te], pred, average="macro")))
    F.assign(family=y, split=df.split.values).to_csv(os.path.join(HERE, "results", f"features_{data}.csv.gz"), index=False)
    json.dump(out, open(os.path.join(HERE, "results", f"features_{data}.json"), "w"), indent=1)
    print(f"{data}: features-only GBM acc={out['features_only_accuracy']:.3f} f1={out['features_only_macro_f1']:.3f}")
    print("top ANOVA features:", list(out["anova_f"])[:8])
    print(F.groupby(y)[["words", "sentences", "newlines", "bullets", "bold", "hedges", "ttr200"]].mean().round(2))


if __name__ == "__main__":
    assert featurize("Sure! Here you go:\n- a\n- b\n**x** `c`")["bullets"] == 2
    main(*sys.argv[1:])
