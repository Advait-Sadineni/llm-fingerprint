"""Training loop + evaluation shared by every neural model."""
import time, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .text import SeqDataset, collate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def loader(seqs, labels, batch, shuffle):
    return DataLoader(SeqDataset(seqs, labels), batch_size=batch, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, dl):
    model.eval()
    probs = []
    for x, _ in dl:
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = model(x.to(DEVICE))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(probs).numpy()


def metrics(y_true, probs, labels):
    y_pred = probs.argmax(1)
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average="macro")),
                per_class_f1=dict(zip(labels, f1_score(y_true, y_pred, average=None, labels=range(len(labels))).tolist())),
                confusion=confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist())


def fit(model, train, val, epochs=15, lr=1e-3, batch=64, patience=3, log=print):
    """train/val = (seqs, labels). Early-stops on val macro-F1; returns (model, history)."""
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    tr_dl, va_dl = loader(*train, batch, True), loader(*val, batch, False)
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * len(y)
        probs = predict(model, va_dl)
        f1 = f1_score(val[1], probs.argmax(1), average="macro")
        acc = accuracy_score(val[1], probs.argmax(1))
        hist.append(dict(epoch=ep, train_loss=tot / len(train[0]), val_acc=float(acc), val_macro_f1=float(f1), secs=time.time() - t0))
        log(f"  ep{ep:02d} loss={hist[-1]['train_loss']:.3f} val_acc={acc:.3f} val_f1={f1:.3f} ({hist[-1]['secs']:.0f}s)")
        if f1 > best:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist
