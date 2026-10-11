"""One CLI for every experiment. Writes results/<name>.json (+ runs/<name>.pt for neural models).

  python run.py --data uf --model cnn --view output
  python run.py --data uf --model lstm --view both --seed 1
  python run.py --data uf --model cnn --train-src flan,sharegpt,evol_instruct,ultrachat,false_qa --test-src truthful_qa
  python run.py --data uf --model cnn --ablate strip_format
"""
import argparse, json, os, random, re, time
import numpy as np, pandas as pd, torch
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from src import text as T
from src.models import build
from src.train import fit, predict, loader, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["evol_instruct", "false_qa", "flan", "sharegpt", "truthful_qa", "ultrachat"]
MD_RE = re.compile(r"(^|\n)\s*(?:[-*+•]|\d+[.)]|#{1,6})\s+|[*_`>#|~]+")


SPECIAL_RE = re.compile(r"<\|[^|>]*\|>")                       # leaked chat-template tokens such as <|close|>
DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "’": "'", "“": '"', "”": '"'})


def ablate_text(txt, how):
    if how == "strip_format":
        return re.sub(r"\s+", " ", MD_RE.sub(" ", txt)).strip()
    if how == "strip_special":                                  # remove template tokens and ASCII-fold typographic punctuation
        return SPECIAL_RE.sub(" ", txt).translate(DASHES)
    return txt


def ablate_tokens(toks, how, rng):
    if how == "cap100":
        return toks[:100]
    if how == "first50":
        return toks[:50]
    if how == "function_words":
        return [t if (t.lower() in ENGLISH_STOP_WORDS or not t[0].isalnum()) else "<w>" for t in toks]
    if how == "shuffle":
        toks = list(toks); rng.shuffle(toks); return toks
    return toks


def load_split(args):
    df = pd.read_csv(os.path.join(HERE, "data", args.data + ".csv.gz"), keep_default_na=False)
    label_col = "model" if args.data == "uf17" else "family"
    train_src = args.train_src.split(",") if args.train_src else SOURCES
    test_src = args.test_src.split(",") if args.test_src else SOURCES
    tr = df[(df.split == "train") & df.source.isin(train_src)]
    va = df[(df.split == "val") & df.source.isin(train_src)]
    if set(test_src) & set(train_src):
        te = df[(df.split == "test") & df.source.isin(test_src)]
    else:                                   # held-out domain: every row of it is unseen, use them all
        te = df[df.source.isin(test_src)]
    if args.frac < 1:
        tr = tr.sample(frac=args.frac, random_state=args.seed)
    if args.smoke:
        tr, va, te = tr.sample(800, random_state=0), va.sample(200, random_state=0), te.sample(200, random_state=0)
    labels = sorted(df[label_col].unique())
    return tr, va, te, labels, label_col


def parse_cfg(cfg):
    out = {}
    for kv in filter(None, cfg.split(",")):
        k, v = kv.split("=")
        out[k] = tuple(int(x) for x in v.split("-")) if "-" in v else int(v)
    return out


def run_name(args):
    parts = [args.data, args.model + ("+" + args.tag if args.tag else ""), args.view]
    if args.cfg:
        parts.append("cfg-" + args.cfg.replace(",", "_").replace("=", ""))
    if args.train_src:
        parts.append("tr-" + args.train_src.replace(",", "+"))
    if args.test_src:
        parts.append("te-" + args.test_src)
    if args.ablate != "none":
        parts.append(args.ablate)
    if args.frac < 1:
        parts.append(f"f{args.frac}")
    parts.append(f"s{args.seed}")
    if args.smoke:
        parts.append("smoke")
    return "_".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="uf"); p.add_argument("--model", default="cnn")
    p.add_argument("--view", default="output", choices=["input", "output", "both"])
    p.add_argument("--train-src"); p.add_argument("--test-src")
    p.add_argument("--ablate", default="none"); p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--smoke", action="store_true"); p.add_argument("--name"); p.add_argument("--force", action="store_true")
    p.add_argument("--cfg", default="", help="model kwargs, e.g. emb=300,channels=256,kernels=2-3-4-5 or hidden=256,layers=2")
    p.add_argument("--tag", default="", help="label for a non-default configuration that should appear in the headline tables (e.g. tuned)")
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    name = args.name or run_name(args)
    out_path = os.path.join(HERE, "results", name + ".json")
    if os.path.exists(out_path) and not args.force:
        print(f"== {name}: exists, skipping"); return
    print(f"== {name}")
    tr, va, te, labels, label_col = load_split(args)
    lab = {l: i for i, l in enumerate(labels)}
    ys = [d[label_col].map(lab).tolist() for d in (tr, va, te)]
    # prompts are capped (128 tokens / 700 chars) so a long ShareGPT prompt cannot crowd the response out of the window
    pairs = [[(ablate_text(i, args.ablate)[:700], ablate_text(o, args.ablate)) for i, o in zip(d.instruction, d.response)] for d in (tr, va, te)]
    texts = [[T.view_text(i, o, args.view) for i, o in ps] for ps in pairs]
    t0 = time.time(); hist = []

    if args.model == "tfidf":
        clf = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, lowercase=False, sublinear_tf=True),
                                       TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200000, lowercase=False, sublinear_tf=True)),
                            LogisticRegression(C=5, max_iter=2000))
        clf.fit(texts[0], ys[0])
        probs = clf.predict_proba(texts[2])
        import joblib; os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
        joblib.dump(dict(clf=clf, labels=labels, view=args.view), os.path.join(HERE, "runs", name + ".joblib"))
    elif args.model == "bert":
        from src.bert import run_bert
        probs, hist = run_bert(texts, ys, len(labels), args, log=print)
    else:
        if args.model == "charcnn":
            seqs = [[T.encode_chars(t) for t in ts] for ts in texts]
            vocab_size, itos = len(T.CHARS) + 2, None
        else:
            toks = [[ablate_tokens(T.view_tokens(i, o, args.view), args.ablate, rng) for i, o in ps] for ps in pairs]
            vocab = T.Vocab(toks[0]); vocab_size, itos = len(vocab), vocab.itos
            seqs = [[vocab.encode(t) for t in ts] for ts in toks]
            print(f"  vocab={vocab_size}")
        model = build(args.model, vocab_size, len(labels), **parse_cfg(args.cfg))
        print(f"  params={sum(p.numel() for p in model.parameters()):,} train/val/test={len(tr)}/{len(va)}/{len(te)}")
        model, hist = fit(model, (seqs[0], ys[0]), (seqs[1], ys[1]), epochs=1 if args.smoke else args.epochs)
        probs = predict(model, loader(seqs[2], ys[2], 128, False))
        os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
        torch.save(dict(state=model.state_dict(), model=args.model, itos=itos, labels=labels, view=args.view),
                   os.path.join(HERE, "runs", name + ".pt"))

    m = metrics(np.array(ys[2]), probs, labels)
    out = dict(name=name, args=vars(args), labels=labels, n_train=len(tr), n_val=len(va), n_test=len(te), secs=time.time() - t0, history=hist, **m)
    os.makedirs(os.path.join(HERE, "results", "preds"), exist_ok=True)
    json.dump(out, open(out_path, "w"))
    np.savez_compressed(os.path.join(HERE, "results", "preds", name + ".npz"), probs=probs.astype(np.float32),     # per-example predictions,
                        y=np.array(ys[2], dtype=np.int16), prompt_ids=np.array(te.prompt_id.tolist()))             # read back by src.aggregate
    print(f"  TEST acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} ({out['secs']:.0f}s)")


if __name__ == "__main__":
    main()
