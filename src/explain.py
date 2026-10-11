"""RQ4 explainability on saved checkpoints.
  python -m src.explain runs/uf_cnn_output_s42.pt   -> results/explain_<name>.json  (top n-grams per class from CNN filters)
  python -m src.explain runs/uf_lstm_output_s42.pt  -> results/explain_<name>.json  (gradient x input token saliency, one example per class)
"""
import json, os, sys
from collections import defaultdict
import numpy as np, pandas as pd, torch
from . import text as T
from .models import build

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    ck = torch.load(path, map_location=DEVICE)
    vocab_size = len(ck["itos"]) if ck["itos"] is not None else len(T.CHARS) + 2      # Char-CNN has no word vocab
    model = build(ck["model"], vocab_size, len(ck["labels"])).to(DEVICE)
    model.load_state_dict(ck["state"]); model.eval()
    vocab = T.Vocab([]); vocab.itos = ck["itos"] or []; vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    return model, vocab, ck


def test_rows(ck, n_per_class=400):
    data = os.path.basename(sys.argv[1]).split("_")[0]
    df = pd.read_csv(os.path.join(HERE, "data", data + ".csv.gz"), keep_default_na=False)
    df = df[df.split == "test"]
    return pd.concat(df[df.family == f].head(n_per_class) for f in ck["labels"])


def cnn_ngrams(model, vocab, ck, df, top=15):
    """For every conv filter, find the n-gram that maximally activates it on test data; score (filter, class) by
    the FC weight. Report the strongest n-grams per class, weighted by activation x weight."""
    W = model.fc.weight.detach().cpu().numpy()                      # (C, F)
    scores = defaultdict(lambda: defaultdict(float))
    for inp, out, fam in zip(df.instruction, df.response, df.family):
        toks = T.view_tokens(inp, out, ck["view"])
        ids = vocab.encode(toks); ids += [0] * max(0, 5 - len(ids))
        x = torch.tensor([ids]).to(DEVICE)
        with torch.no_grad():
            e = model.emb(x).transpose(1, 2)
            off = 0
            for conv in model.convs:
                k = conv.kernel_size[0]
                a = torch.relu(conv(e))[0]                           # (channels, L-k+1)
                act, pos = a.max(dim=1)
                for f in range(a.shape[0]):
                    if act[f] <= 0:
                        continue
                    ng = " ".join(toks[pos[f]: pos[f] + k]).replace("\n", "\\n")
                    for c, lab in enumerate(ck["labels"]):
                        scores[lab][ng] += float(act[f]) * float(W[c, off + f])
                off += a.shape[0]
    return {lab: sorted(d.items(), key=lambda kv: -kv[1])[:top] for lab, d in scores.items()}


def lstm_saliency(model, vocab, ck, df):
    """Gradient x input on the embedding for one correctly-classified example per class."""
    model.train()          # cuDNN RNN backward needs train mode; the model has no dropout inside the LSTM, so outputs are unchanged
    out = {}
    for lab_idx, lab in enumerate(ck["labels"]):
        for inp, resp in zip(df[df.family == lab].instruction, df[df.family == lab].response):
            toks = T.view_tokens(inp, resp, ck["view"])[:160]
            x = torch.tensor([vocab.encode(toks)]).to(DEVICE)
            emb = model.emb(x).detach().requires_grad_(True)
            mask = (x != 0).unsqueeze(-1)
            h, _ = model.lstm(emb)
            mx = h.masked_fill(~mask, -1e4).max(dim=1).values
            mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            logits = model.fc(torch.cat([mx, mean], 1))
            if logits.argmax().item() != lab_idx:
                continue
            logits[0, lab_idx].backward()
            sal = (emb.grad * emb).sum(-1)[0].abs().detach().cpu().numpy()
            sal = sal / (sal.max() + 1e-9)
            out[lab] = dict(tokens=toks, saliency=np.round(sal, 3).tolist(), prob=float(torch.softmax(logits, 1)[0, lab_idx]))
            break
    return out


def main(path):
    model, vocab, ck = load(path)
    df = test_rows(ck)
    name = os.path.splitext(os.path.basename(path))[0]
    res = dict(checkpoint=name, model=ck["model"], labels=ck["labels"])
    if ck["model"] == "cnn":
        res["top_ngrams"] = cnn_ngrams(model, vocab, ck, df)
        for lab, ngs in res["top_ngrams"].items():
            print(lab, "->", [n for n, _ in ngs[:10]])
    elif ck["model"] == "lstm":
        res["saliency"] = lstm_saliency(model, vocab, ck, df)
        for lab, d in res["saliency"].items():
            top = sorted(zip(d["saliency"], d["tokens"]), reverse=True)[:8]
            print(lab, f"p={d['prob']:.2f}", "->", [t for _, t in top])
    json.dump(res, open(os.path.join(HERE, "results", f"explain_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
