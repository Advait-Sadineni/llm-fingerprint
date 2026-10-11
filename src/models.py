"""The classifiers: TextCNN (Kim 2014), BiLSTM, Char-CNN (Zhang 2015), CNN-BiLSTM hybrid."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, emb=128, channels=128, kernels=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, channels, k) for k in kernels)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernels), n_classes)

    def features(self, x):                       # (B, L) -> (B, C*K); exposed for filter probing
        e = self.emb(x).transpose(1, 2)          # (B, E, L)
        return torch.cat([F.relu(c(e)).max(dim=2).values for c in self.convs], dim=1)

    def forward(self, x):
        return self.fc(self.drop(self.features(x)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_classes, emb=128, hidden=128, layers=1, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.lstm(self.emb(x))             # (B, L, 2H)
        mx = h.masked_fill(~mask, -1e4).max(dim=1).values
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(torch.cat([mx, mean], dim=1)))


class CNNLSTM(nn.Module):
    """Conv n-gram feature extractor feeding a BiLSTM: the 'combine models' variant."""
    def __init__(self, vocab_size, n_classes, emb=128, channels=128, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.conv = nn.Conv1d(emb, channels, 3, padding=1)
        self.lstm = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        c = F.relu(self.conv(self.emb(x).transpose(1, 2))).transpose(1, 2)   # (B, L, C)
        h, _ = self.lstm(c)
        return self.fc(self.drop(h.max(dim=1).values))


class CharCNN(nn.Module):
    """Zhang et al. 2015 (small): 6 conv layers over characters, input length 1024."""
    def __init__(self, n_chars, n_classes, channels=256, dropout=0.5, length=1024):
        super().__init__()
        self.emb = nn.Embedding(n_chars, 32, padding_idx=0)
        layers, inp = [], 32
        for k, pool in [(7, 3), (7, 3), (3, 0), (3, 0), (3, 0), (3, 3)]:
            layers += [nn.Conv1d(inp, channels, k), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool1d(pool))
            inp = channels
        self.convs = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.convs(torch.zeros(1, 32, length)).numel()
        self.fc = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, n_classes))
        self.length = length

    def forward(self, x):
        x = F.pad(x, (0, self.length - x.shape[1]))[:, :self.length]
        return self.fc(self.convs(self.emb(x).transpose(1, 2)).flatten(1))


def build(name, vocab_size, n_classes, **kw):
    return {"cnn": TextCNN, "lstm": BiLSTM, "cnnlstm": CNNLSTM, "charcnn": CharCNN}[name](vocab_size, n_classes, **kw)


if __name__ == "__main__":
    x = torch.randint(1, 100, (4, 50))
    for name in ["cnn", "lstm", "cnnlstm", "charcnn"]:
        assert build(name, 100, 5)(x).shape == (4, 5), name
    print("models.py ok")
