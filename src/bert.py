"""DistilBERT fine-tuning: the transformer upper bound for the model ladder."""
import time, copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NAME = "distilbert-base-cased"      # cased: capitalisation is part of the fingerprint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode(tok, texts, max_len=512):
    enc = tok(texts, truncation=True, max_length=max_len, padding=False)
    return [dict(input_ids=i, attention_mask=a) for i, a in zip(enc["input_ids"], enc["attention_mask"])]


def make_loader(items, ys, tok, batch, shuffle):
    def collate(b):
        feats, labels = zip(*b)
        pad = tok.pad(list(feats), return_tensors="pt")
        return pad["input_ids"], pad["attention_mask"], torch.tensor(labels)
    return DataLoader(list(zip(items, ys)), batch_size=batch, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, dl):
    model.eval(); out = []
    for ids, mask, _ in dl:
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits
        out.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(out).numpy()


def run_bert(texts, ys, n_classes, args, log=print, epochs=3, lr=2e-5, batch=16):
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=n_classes).to(DEVICE)
    items = [encode(tok, t) for t in texts]
    tr, va, te = (make_loader(items[i], ys[i], tok, batch if i == 0 else 64, i == 0) for i in range(3))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    best, best_state, hist = -1, None, []
    for ep in range(1, (1 if args.smoke else epochs) + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for ids, mask, y in tr:
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                loss = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE), labels=y.to(DEVICE)).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item() * len(y)
        probs = predict(model, va)
        f1, acc = f1_score(ys[1], probs.argmax(1), average="macro"), accuracy_score(ys[1], probs.argmax(1))
        hist.append(dict(epoch=ep, train_loss=tot / len(ys[0]), val_acc=float(acc), val_macro_f1=float(f1), secs=time.time() - t0))
        log(f"  ep{ep:02d} loss={hist[-1]['train_loss']:.3f} val_acc={acc:.3f} val_f1={f1:.3f} ({hist[-1]['secs']:.0f}s)")
        if f1 > best:
            best, best_state = f1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return predict(model, te), hist
