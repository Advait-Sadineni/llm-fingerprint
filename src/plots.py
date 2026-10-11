"""All report figures, regenerated from results/*.json. `python -m src.plots` skips figures whose inputs are missing."""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
# categorical palette (fixed slot order, never cycled) + one-hue sequential ramp
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", green="#008300", violet="#4a3aa7", red="#e34948")
MODEL_COLOR = {"tfidf": C["violet"], "cnn": C["blue"], "lstm": C["orange"], "cnn+tuned": C["blue"], "lstm+tuned": C["orange"],
               "cnnlstm": C["aqua"], "charcnn": C["yellow"], "bert": C["magenta"], "ensemble": C["green"]}
MODEL_NAME = {"tfidf": "TF-IDF + LR", "cnn": "TextCNN", "lstm": "BiLSTM", "cnn+tuned": "TextCNN (tuned)", "lstm+tuned": "BiLSTM (tuned)",
              "cnnlstm": "CNN-BiLSTM", "charcnn": "Char-CNN", "bert": "DistilBERT", "ensemble": "Ensemble"}
MODELS = list(MODEL_COLOR)
VIEW_COLOR = {"input": "#86b6ef", "output": "#2a78d6", "both": "#104281"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#3987e5", "#104281"])
INK, MUTED = "#1f2328", "#6e7781"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e8eb", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight"})


def summary():
    return pd.read_csv(os.path.join(HERE, "results", "summary.csv"), keep_default_na=False)


def run(name):
    p = os.path.join(HERE, "results", name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def bar_models(ax, df, title, ylabel="Macro-F1"):
    df = df.set_index("model").reindex([m for m in MODELS if m in set(df.model)])
    x = np.arange(len(df))
    ax.bar(x, df.f1_mean, yerr=df.f1_std, color=[MODEL_COLOR[m] for m in df.index], width=0.62, capsize=2, error_kw=dict(lw=0.8, ecolor=INK))
    for i, v in enumerate(df.f1_mean):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, [MODEL_NAME[m].replace(" (", chr(10) + "(") for m in df.index], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="x", visible=False)


def fig_rq1():
    s = summary(); s = s[(s.view == "output") & (s.train_src == "") & (s.ablate == "none") & (s.frac == 1.0)]
    datasets = [d for d in ["uf", "modern", "uf17"] if d in set(s.data)]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.1), squeeze=False)
    titles = {"uf": "UltraFeedback (2023), {n} families", "modern": "Modern set (2026), {n} families", "uf17": "UltraFeedback, {n} models"}
    for ax, d in zip(axes[0], datasets):
        n_cls = len(run(f"{d}_cnn_output_s42")["labels"])
        bar_models(ax, s[s.data == d], titles[d].format(n=n_cls))
        chance = 1 / n_cls
        ax.axhline(chance, color=MUTED, lw=0.8, ls="--"); ax.text(len(s[s.data == d]) - 0.5, chance + 0.01, "chance", fontsize=7, color=MUTED, ha="right")
    fig.savefig(os.path.join(FIG, "rq1_models.png")); plt.close(fig)


def fig_rq2():
    s = summary(); s = s[(s.train_src == "") & (s.ablate == "none") & (s.frac == 1.0)]
    datasets = [d for d in ["uf", "uf_natural", "modern"] if d in set(s.data)]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.8 * len(datasets), 3.1), squeeze=False)
    titles = {"uf": "UltraFeedback (controlled source mix)", "uf_natural": "UltraFeedback (natural source mix)", "modern": "Modern set (same prompts for all)"}
    for ax, d in zip(axes[0], datasets):
        t = s[s.data == d]; models = [m for m in MODELS if m in set(t.model) and "+" not in m and m != "ensemble"]
        x = np.arange(len(models)); w = 0.26
        for j, v in enumerate(["input", "output", "both"]):
            tv = t[t.view == v].set_index("model").reindex(models)
            ax.bar(x + (j - 1) * w, tv.f1_mean, w - 0.03, yerr=tv.f1_std, color=VIEW_COLOR[v], label=v, capsize=1.5, error_kw=dict(lw=0.6, ecolor=INK))
        n_cls = len(run(f"{d}_cnn_output_s42")["labels"])
        ax.axhline(1 / n_cls, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(x, [MODEL_NAME[m] for m in models], rotation=20, ha="right"); ax.set_ylim(0, 1)
        ax.set_title(titles[d], loc="left", fontsize=10); ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Macro-F1"); axes[0][0].legend(title="view", fontsize=8, loc="upper left")
    fig.savefig(os.path.join(FIG, "rq2_views.png")); plt.close(fig)


def confusion(ax, r, title):
    cm = np.array(r["confusion"], dtype=float); cm = cm / cm.sum(1, keepdims=True)
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1)
    labs = r["labels"]
    ax.set_xticks(range(len(labs)), labs, rotation=45, ha="right", fontsize=7 if len(labs) > 8 else 8)
    ax.set_yticks(range(len(labs)), labs, fontsize=7 if len(labs) > 8 else 8)
    if len(labs) <= 8:
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cm[i, j] > 0.5 else INK)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, loc="left", fontsize=10); ax.grid(False)


def fig_confusions():
    for data in ["uf", "modern"]:
        rs = [(m, run(f"{data}_{m}_output_s42")) for m in ["cnn", "lstm", "bert"]]
        rs = [(m, r) for m, r in rs if r]
        if not rs:
            continue
        fig, axes = plt.subplots(1, len(rs), figsize=(3.4 * len(rs), 3.2), squeeze=False)
        for k, (ax, (m, r)) in enumerate(zip(axes[0], rs)):
            confusion(ax, r, f"{MODEL_NAME[m]}  acc={r['accuracy']:.2f}")
            if k:
                ax.set_ylabel("")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"confusion_{data}.png")); plt.close(fig)
    r = run("uf17_cnn_output_s42") or run("uf17_bert_output_s42")
    if r:
        fig, ax = plt.subplots(figsize=(5.2, 4.8)); confusion(ax, r, f"16-way model identification ({MODEL_NAME[r['args']['model']]}, acc={r['accuracy']:.2f})")
        fig.savefig(os.path.join(FIG, "confusion_uf17.png")); plt.close(fig)


def fig_training_curves():
    rs = [(m, run(f"uf_{m}_output_s42")) for m in ["cnn", "lstm", "cnnlstm", "charcnn", "bert"]]
    rs = [(m, r) for m, r in rs if r and r["history"]]
    if not rs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for m, r in rs:
        h = pd.DataFrame(r["history"])
        axes[0].plot(h.epoch, h.train_loss, color=MODEL_COLOR[m], lw=1.6, label=MODEL_NAME[m])
        axes[1].plot(h.epoch, h.val_macro_f1, color=MODEL_COLOR[m], lw=1.6, label=MODEL_NAME[m])
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val macro-F1")
    axes[1].legend(fontsize=7.5); fig.savefig(os.path.join(FIG, "training_curves.png")); plt.close(fig)


def fig_loso():
    s = summary(); s = s[(s.train_src != "") & (s.view == "output")]
    if s.empty:
        return
    ind = summary(); ind = ind[(ind.train_src == "") & (ind.view == "output") & (ind.ablate == "none") & (ind.frac == 1.0)]
    datasets = [d for d in ["uf", "modern"] if d in set(s.data)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.0), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        t = s[s.data == d]; models = [m for m in MODELS if m in set(t.model)]
        srcs = ["evol_instruct", "false_qa", "flan", "sharegpt", "truthful_qa", "ultrachat"]
        M = np.array([[t[(t.model == m) & (t.test_src == src)].f1_mean.mean() if len(t[(t.model == m) & (t.test_src == src)]) else np.nan for src in srcs] for m in models])
        ax.imshow(M, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        for i in range(len(models)):
            for j in range(len(srcs)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if M[i, j] > 0.55 else INK)
        ylabels = []
        for m in models:
            base = ind[(ind.data == d) & (ind.model == m)].f1_mean
            ylabels.append(f"{MODEL_NAME[m]}\n(in-domain {base.iloc[0]:.2f})" if len(base) else MODEL_NAME[m])
        ax.set_yticks(range(len(models)), ylabels, fontsize=7.5); ax.set_xticks(range(len(srcs)), srcs, rotation=30, ha="right", fontsize=7.5)
        ax.set_title(f"Held-out source macro-F1 ({'UltraFeedback' if d == 'uf' else 'modern set'})", loc="left", fontsize=10); ax.grid(False)
    fig.savefig(os.path.join(FIG, "rq3_loso.png")); plt.close(fig)


def fig_curves():
    s = summary(); s = s[(s.view == "output") & (s.train_src == "") & (s.ablate == "none") & (s.data == "uf") & s.model.isin(["cnn", "lstm"])]
    if s.frac.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for m in ["cnn", "lstm"]:
        t = s[s.model == m].sort_values("frac")
        n = t.frac * 16015
        ax.plot(n, t.f1_mean, "-o", color=MODEL_COLOR[m], lw=1.6, ms=4, label=MODEL_NAME[m])
        ax.fill_between(n, t.f1_mean - t.f1_std, t.f1_mean + t.f1_std, color=MODEL_COLOR[m], alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_xlabel("training examples"); ax.set_ylabel("test macro-F1"); ax.legend(fontsize=8)
    ax.set_title("Learning curve (UltraFeedback, output view)", loc="left", fontsize=10)
    fig.savefig(os.path.join(FIG, "rq3_learning_curve.png")); plt.close(fig)


def fig_ablation():
    s = summary(); s = s[(s.view == "output") & (s.train_src == "") & (s.data == "uf") & (s.frac == 1.0) & s.model.isin(["cnn", "lstm"])]
    abl = [a for a in ["strip_format", "cap100", "first50", "function_words", "shuffle"] if a in set(s.ablate)]
    if not abl:
        return
    names = {"strip_format": "markdown/newlines\nstripped", "cap100": "first 100\ntokens", "first50": "first 50\ntokens", "function_words": "function words\nonly", "shuffle": "word order\nshuffled"}
    fig, ax = plt.subplots(figsize=(5.4, 3.0)); x = np.arange(len(abl)); w = 0.36
    for j, m in enumerate(["cnn", "lstm"]):
        base = s[(s.model == m) & (s.ablate == "none")].f1_mean.iloc[0]
        t = s[(s.model == m) & (s.ablate != "none")].set_index("ablate").reindex(abl)
        ax.bar(x + (j - 0.5) * w, t.f1_mean - base, w - 0.04, yerr=t.f1_std, color=MODEL_COLOR[m], label=f"{MODEL_NAME[m]} (full = {base:.3f})", capsize=1.5, error_kw=dict(lw=0.6, ecolor=INK))
    ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(x, [names[a] for a in abl], fontsize=8); ax.set_ylabel("Δ macro-F1 vs full text"); ax.legend(fontsize=8)
    ax.set_title("What the classifiers lose when a signal is removed", loc="left", fontsize=10); ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "rq4_ablation.png")); plt.close(fig)


def fig_stylometry():
    for data in ["uf", "modern"]:
        p = os.path.join(HERE, "results", f"features_{data}.csv.gz")
        if not os.path.exists(p):
            continue
        F = pd.read_csv(p); fams = sorted(F.family.unique())
        feats = [("words", "words per response"), ("newlines", "line breaks"), ("ttr200", "type-token ratio (first 200 words)"), ("bullets", "bullet lines"), ("bold", "**bold** spans"), ("hedges", "hedge/refusal phrases")]
        fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.6))
        for ax, (f, title) in zip(axes.flat, feats):
            vals = [F[F.family == fam][f] for fam in fams]
            if f in ("bullets", "bold", "hedges"):
                means = [v.mean() for v in vals]
                ax.bar(range(len(fams)), means, color="#2a78d6", width=0.6)
                ax.set_ylabel("mean per response")
            else:
                bp = ax.boxplot(vals, positions=range(len(fams)), widths=0.55, showfliers=False, patch_artist=True)
                for b in bp["boxes"]:
                    b.set(facecolor="#cde2fb", edgecolor="#2a78d6", linewidth=1)
                for k in ("whiskers", "caps", "medians"):
                    for l in bp[k]:
                        l.set(color="#104281", linewidth=1)
            ax.set_xticks(range(len(fams)), fams, rotation=25, ha="right", fontsize=7.5); ax.set_title(title, loc="left", fontsize=9); ax.grid(axis="x", visible=False)
        fig.suptitle(f"Stylometric profile per family ({'UltraFeedback' if data == 'uf' else 'modern set'})", x=0.01, ha="left", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"rq4_stylometry_{data}.png")); plt.close(fig)


def fig_saliency():
    for data in ["uf", "modern"]:
        r = run(f"explain_{data}_lstm_output_s42")
        if not r or "saliency" not in r:
            continue
        labs = list(r["saliency"])
        fig, axes = plt.subplots(len(labs), 1, figsize=(8.5, 1.05 * len(labs)))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, hspace=0.1)
        for ax, lab in zip(np.atleast_1d(axes), labs):
            d = r["saliency"][lab]; ax.axis("off")
            ax.text(0, 0.95, f"{lab}  (p = {d['prob']:.2f})", fontsize=8, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
            x, y = 0.0, 0.55
            for tok, s in zip(d["tokens"][:70], d["saliency"][:70]):
                t = "⏎" if tok == "\n" else tok
                ax.text(x, y, t, fontsize=7.5, color=INK, va="top", transform=ax.transAxes,
                        bbox=dict(boxstyle="square,pad=0.12", fc=(0.165, 0.47, 0.84, 0.08 + 0.85 * s), ec="none"))
                x += 0.0115 * max(len(t), 1) + 0.008
                if x > 0.98:
                    x, y = 0.0, y - 0.45
                if y < -0.2:
                    break
        fig.suptitle("BiLSTM gradient×input saliency (darker = more influential for the true class)", x=0.01, ha="left", fontsize=9)
        fig.savefig(os.path.join(FIG, f"rq4_saliency_{data}.png")); plt.close(fig)


def fig_paraphrase():
    r = run("paraphrase_uf")
    if not r:
        return
    models = [m for m in MODELS if m in r["before"]]
    fig, ax = plt.subplots(figsize=(4.6, 2.9)); x = np.arange(len(models)); w = 0.36
    ax.bar(x - w / 2, [r["before"][m] for m in models], w - 0.04, color=[MODEL_COLOR[m] for m in models], label="original")
    ax.bar(x + w / 2, [r["after"][m] for m in models], w - 0.04, color=[MODEL_COLOR[m] for m in models], alpha=0.45, hatch="///", label="paraphrased")
    ax.axhline(1 / len(r["labels"]), color=MUTED, lw=0.8, ls="--")
    ax.set_xticks(x, [MODEL_NAME[m] for m in models], rotation=20, ha="right"); ax.set_ylabel("accuracy on the 300 attacked responses"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.set_title(f"Paraphrase attack (rewritten by {r['paraphraser']})", loc="left", fontsize=10); ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(FIG, "rq4_paraphrase.png")); plt.close(fig)


def fig_crossgen():
    r = run("crossgen")
    if not r:
        return
    M = pd.DataFrame(r["uf_model_on_modern"]).T   # rows: modern family, cols: predicted UF family
    fig, ax = plt.subplots(figsize=(4.6, 3.2)); ax.imshow(M.values, cmap=BLUES, vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if M.values[i, j] > 0.55 else INK)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=30, ha="right"); ax.set_yticks(range(M.shape[0]), M.index)
    ax.set_xlabel("predicted 2023 family"); ax.set_ylabel("actual 2026 family"); ax.grid(False)
    ax.set_title(f"Cross-generation attribution ({MODEL_NAME[r['model']]})", loc="left", fontsize=10)
    fig.savefig(os.path.join(FIG, "rq3_crossgen.png")); plt.close(fig)


if __name__ == "__main__":
    for f in [fig_rq1, fig_rq2, fig_confusions, fig_training_curves, fig_loso, fig_curves, fig_ablation, fig_stylometry, fig_saliency, fig_paraphrase, fig_crossgen]:
        try:
            f(); print("ok  ", f.__name__)
        except Exception as e:
            print("FAIL", f.__name__, repr(e))
    print(sorted(os.listdir(FIG)))
