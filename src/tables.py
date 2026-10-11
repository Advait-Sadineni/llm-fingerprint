"""Generate typst table snippets in report/tables/ from results/summary.csv + stats, so every number in the PDF is
produced by code. `python -m src.tables`"""
import json, os
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "report", "tables")
os.makedirs(OUT, exist_ok=True)
NAME = {"tfidf": "TF-IDF + LR", "cnn": "TextCNN", "lstm": "BiLSTM", "cnnlstm": "CNN-BiLSTM", "charcnn": "Char-CNN", "bert": "DistilBERT",
        "cnn+tuned": "TextCNN (tuned)", "lstm+tuned": "BiLSTM (tuned)", "ensemble": "Ensemble of four"}
ORDER = ["tfidf", "cnn", "lstm", "cnn+tuned", "lstm+tuned", "cnnlstm", "charcnn", "bert", "ensemble"]
SRC = ["evol_instruct", "false_qa", "flan", "sharegpt", "truthful_qa", "ultrachat"]


def ms(m, s, bold=False):
    if pd.isna(m):
        return "—"
    txt = f"{m:.3f}" if pd.isna(s) or s < 0.0005 else f"{m:.3f} ± {s:.3f}"
    return f"*{txt}*" if bold else txt


def write(name, header, rows, align=None):
    cols = len(header)
    align = align or "(left," + "right," * (cols - 1) + ")"
    cells = ", ".join(f"[{h}]" for h in header) + ",\n" + ",\n".join("    " + ", ".join(f"[{c}]" for c in r) for r in rows)
    open(os.path.join(OUT, name + ".typ"), "w", encoding="utf-8").write(f"#table(columns: {cols}, align: {align},\n    {cells},\n)\n")


def views_table(s, data, name, views=("output", "input", "both")):
    t = s[(s.data == data) & (s.train_src == "") & (s.ablate == "none") & (s.frac == 1.0)]
    rows = []
    best = {v: t[t.view == v].f1_mean.max() for v in views}
    for m in ORDER:
        r = [NAME[m]]
        for v in views:
            x = t[(t.model == m) & (t.view == v)]
            if len(x):
                r += [ms(x.acc_mean.iloc[0], x.acc_std.iloc[0]), ms(x.f1_mean.iloc[0], x.f1_std.iloc[0], bold=x.f1_mean.iloc[0] == best[v])]
            else:
                r += ["—", "—"]
        if any(c != "—" for c in r[1:]):
            rows.append(r)
    header = ["*Model*"] + [f"*{v.capitalize()} acc.*" if i % 2 == 0 else f"*{v.capitalize()} F1*" for v in views for i in (0, 1)]
    write(name, header, rows)


def main():
    s = pd.read_csv(os.path.join(HERE, "results", "summary.csv"), keep_default_na=False)
    stats = json.load(open(os.path.join(HERE, "results", "stats.json")))
    views_table(s, "uf", "uf_views")
    views_table(s, "uf_natural", "uf_natural_views")
    if "modern" in set(s.data):
        views_table(s, "modern", "modern_views")
    # fine-grained
    t = s[(s.data == "uf17") & (s.view == "output")]
    write("uf17", ["*Model*", "*Accuracy*", "*Macro-F1*"], [[NAME[m], ms(t[t.model == m].acc_mean.iloc[0], 0), ms(t[t.model == m].f1_mean.iloc[0], 0)] for m in ORDER if m in set(t.model)])
    # LOSO
    for data in ["uf", "modern"]:
        t = s[(s.data == data) & (s.test_src != "")]
        if t.empty:
            continue
        ind = s[(s.data == data) & (s.train_src == "") & (s.view == "output") & (s.ablate == "none") & (s.frac == 1.0)]
        rows = []
        for m in ["tfidf", "cnn", "lstm"]:
            if m not in set(t.model):
                continue
            r = [NAME[m], ms(ind[ind.model == m].f1_mean.iloc[0], 0)]
            vals = [t[(t.model == m) & (t.test_src == src)].f1_mean for src in SRC]
            r += [ms(v.iloc[0], 0) if len(v) else "—" for v in vals]
            r.append(ms(pd.Series([v.iloc[0] for v in vals if len(v)]).mean(), 0))
            rows.append(r)
        write(f"loso_{data}", ["*Model*", "*In-domain*"] + [f"*{x.replace('_', '-')}*" for x in SRC] + ["*Mean held-out*"], rows)
    # ablations
    t = s[(s.data == "uf") & (s.view == "output") & (s.train_src == "") & (s.frac == 1.0)]
    names = {"none": "Full text", "strip_format": "Markdown and line breaks stripped", "cap100": "First 100 tokens", "first50": "First 50 tokens", "function_words": "Function words, punctuation and line breaks only", "shuffle": "Word order shuffled"}
    rows = []
    for a in ["none", "strip_format", "cap100", "first50", "function_words", "shuffle"]:
        r = [names[a]]
        for m in ["cnn", "lstm"]:
            x = t[(t.model == m) & (t.ablate == a)]
            base = t[(t.model == m) & (t.ablate == "none")].f1_mean.iloc[0]
            r.append(ms(x.f1_mean.iloc[0], x.f1_std.iloc[0]) + ("" if a == "none" else f" ({x.f1_mean.iloc[0] - base:+.3f})") if len(x) else "—")
        rows.append(r)
    write("ablation", ["*Text seen by the classifier*", "*TextCNN F1*", "*BiLSTM F1*"], rows)
    # learning curve
    t = s[(s.data == "uf") & (s.view == "output") & (s.ablate == "none") & (s.train_src == "") & s.model.isin(["cnn", "lstm"])]
    rows = []
    for f in sorted(t.frac.unique()):
        r = [f"{int(round(f * 16015)):,} ({int(f * 100)}%)"]
        for m in ["cnn", "lstm"]:
            x = t[(t.model == m) & (t.frac == f)]
            r.append(ms(x.f1_mean.iloc[0], x.f1_std.iloc[0]))
        rows.append(r)
    write("curve", ["*Training examples*", "*TextCNN F1*", "*BiLSTM F1*"], rows)
    # paraphrase
    p = os.path.join(HERE, "results", "paraphrase_uf.json")
    if os.path.exists(p):
        pr = json.load(open(p))
        rows = [[NAME[m], f"{pr['before'][m]:.3f}", f"{pr['after'][m]:.3f}", f"{pr['after'][m] - pr['before'][m]:+.3f}",
                 max(pr["after_pred_dist"][m], key=pr["after_pred_dist"][m].get) + f" ({max(pr['after_pred_dist'][m].values()):.2f})"] for m in ORDER if m in pr["before"]]
        write("paraphrase", ["*Model*", "*Original*", "*Paraphrased*", "*Δ*", "*Most predicted after*"], rows)
    # typographic / template artifacts per family (share of responses containing each pattern)
    pats = [("Markdown header", r"(?:^|\n)#{1,6} "), ("Horizontal rule (three hyphens)", r"\n-{3,}\n"), ("Bold (double asterisks)", r"\*\*"), ("En dash U+2013", "–"),
            ("Em dash U+2014", "—"), ("Non-breaking hyphen U+2011", "‑"), ("Template token <|…|>", r"<\|[^|>]*\|>")]
    for data in ["uf", "modern"]:
        p = os.path.join(HERE, "data", data + ".csv.gz")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, keep_default_na=False)
        fams = sorted(df.family.unique())
        rows = [[name] + [f"{df[df.family == f].response.str.contains(pat, regex=True).mean():.2f}" for f in fams] for name, pat in pats]
        write(f"artifacts_{data}", ["*Pattern*"] + [f"*{f}*" for f in fams], rows)
    # modern strip_special ablation
    t = s[(s.data == "modern") & (s.view == "output") & (s.train_src == "") & (s.frac == 1.0) & s.model.isin(["cnn", "lstm"])]
    if "strip_special" in set(t.ablate):
        rows = []
        for m in ["cnn", "lstm"]:
            base = t[(t.model == m) & (t.ablate == "none")]; ab = t[(t.model == m) & (t.ablate == "strip_special")]
            rows.append([NAME[m], ms(base.f1_mean.iloc[0], base.f1_std.iloc[0]), ms(ab.f1_mean.iloc[0], ab.f1_std.iloc[0]), f"{ab.f1_mean.iloc[0] - base.f1_mean.iloc[0]:+.3f}"])
        write("modern_strip_special", ["*Model*", "*Full text*", "*Template tokens removed, punctuation ASCII-folded*", "*Δ*"], rows)
    # hyper-parameter sweep (validation only)
    sw = pd.read_csv(os.path.join(HERE, "results", "sweep.csv"))
    if len(sw):
        rows = []
        for m in ["cnn", "lstm"]:
            t = sw[sw.model == m].sort_values("val_f1", ascending=False)
            for _, r in t.iterrows():
                rows.append([NAME[m], r.cfg.replace(",", ", ").replace("=", " = ").replace("-", ","), f"{r.val_f1:.3f}"])
        write("sweep", ["*Model*", "*Configuration*", "*Val macro-F1*"], rows, align="(left, left, right)")
    # stats
    lines = []
    for k, v in stats.items():
        lines.append(f"{k}: {json.dumps(v)}")
    open(os.path.join(OUT, "stats.txt"), "w").write("\n".join(lines))
    # per-class F1 for uf cnn/lstm/bert (seed 42)
    rows = []
    for m in ["cnn", "lstm", "bert"]:
        f = os.path.join(HERE, "results", f"uf_{m}_output_s42.json")
        if os.path.exists(f):
            r = json.load(open(f)); rows.append([NAME[m]] + [f"{r['per_class_f1'][l]:.2f}" for l in r["labels"]])
    if rows:
        write("per_class_uf", ["*Model*"] + [f"*{l}*" for l in json.load(open(os.path.join(HERE, "results", "uf_cnn_output_s42.json")))["labels"]], rows)
    # README results table (markdown), spliced between the RESULTS_TABLE markers
    head = s[(s.view == "output") & (s.train_src == "") & (s.ablate == "none") & (s.frac == 1.0)]
    datasets = [d for d in ["uf", "modern", "uf17"] if d in set(head.data)]
    md = ["| Model | " + " | ".join({"uf": "UltraFeedback 2023 (5 families)", "modern": "Modern 2026 (5 families)", "uf17": "UltraFeedback (16 models)"}[d] for d in datasets) + " |",
          "|---|" + "---:|" * len(datasets)]
    for m in ORDER:
        cells = []
        for d in datasets:
            x = head[(head.data == d) & (head.model == m)]
            cells.append(ms(x.f1_mean.iloc[0], x.f1_std.iloc[0]).replace("*", "**") if len(x) else "—")
        if any(c != "—" for c in cells):
            md.append(f"| {NAME[m]} | " + " | ".join(cells) + " |")
    chance = " · ".join(f"{d}: {1 / len(json.load(open(os.path.join(HERE, 'results', f'{d}_cnn_output_s42.json')))['labels']):.3f}" for d in datasets)
    NL = chr(10)
    md.append(NL + f"Test macro-F1, mean ± std over 3 seeds where available (response text only). Chance: {chance}.")
    readme = os.path.join(HERE, "README.md"); r = open(readme, encoding="utf-8").read()
    start, end = "<!-- RESULTS_TABLE -->", "<!-- /RESULTS_TABLE -->"
    block = start + NL + NL.join(md) + NL + end
    r = r.replace(start, block, 1) if end not in r else r[: r.index(start)] + block + r[r.index(end) + len(end):]
    open(readme, "w", encoding="utf-8", newline=NL).write(r)
    print("tables:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
