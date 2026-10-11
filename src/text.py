"""Tokenizer, vocab, and PyTorch datasets. Case, punctuation and newlines are kept as tokens
because formatting is part of an LLM's fingerprint."""
import re, string
from collections import Counter
import torch
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"\n|[A-Za-z]+(?:'[a-z]+)?|\d+|[^\sA-Za-z\d]")
PAD, UNK, SEP = 0, 1, 2
MAX_IN, MAX_OUT = 128, 512
CHARS = "\n" + string.printable[:95]            # printable ascii + newline; idx 0 = pad, 1 = unk
CHAR_IDX = {c: i + 2 for i, c in enumerate(CHARS)}
MAX_CHARS = 1024


def tokenize(s):
    return TOKEN_RE.findall(s)


class Vocab:
    def __init__(self, token_lists, max_size=30000, min_freq=2):
        cnt = Counter(t for toks in token_lists for t in toks)
        self.itos = ["<pad>", "<unk>", "<sep>"] + [t for t, c in cnt.most_common(max_size) if c >= min_freq]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(t, UNK) for t in toks]


def view_tokens(inp, out, view):
    """Return the token list for a view: input / output / both (input <sep> output)."""
    if view == "input":
        return tokenize(inp)[:MAX_IN]
    if view == "output":
        return tokenize(out)[:MAX_OUT]
    return tokenize(inp)[:MAX_IN] + ["<sep>"] + tokenize(out)[:MAX_OUT]


def view_text(inp, out, view):
    return inp if view == "input" else out if view == "output" else inp + "\n<sep>\n" + out


def encode_chars(text):
    return [CHAR_IDX.get(c, 1) for c in text[:MAX_CHARS]]


class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs, self.labels = seqs, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]


def collate(batch):
    seqs, labels = zip(*batch)
    L = max(5, max(len(s) for s in seqs))        # min 5 so the k=5 conv always fits
    x = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x, torch.tensor(labels)


if __name__ == "__main__":
    toks = tokenize("Hello, world!\nIt's 2026.")
    assert toks == ["Hello", ",", "world", "!", "\n", "It's", "2026", "."], toks
    v = Vocab([toks, toks], min_freq=1)
    assert v.encode(["Hello", "zzz"]) == [v.stoi["Hello"], UNK]
    x, y = collate([([3, 4], 0), ([5], 1)])
    assert x.shape == (2, 5) and y.tolist() == [0, 1]
    assert encode_chars("a\n")[1] == CHAR_IDX["\n"]
    print("text.py ok")
