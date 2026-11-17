"""Small student networks that see the 34 hand-made features PLUS a compact raw-book window.

Motivation (results/distill_results.md): DeepLOB's edge comes from raw-book patterns the 34 features cannot
express, so a features-only student cannot copy it. These students also get the last K = 10 book updates of
the top L = 5 levels (K x 4L = 200 values): prices as ticks relative to the current mid, sizes z-scored as for
DeepLOB. Both stay small enough for ~10 us CPU inference.

  RawMLP   features + flattened raw window -> 64 -> 32 -> 2                       (~17k params)
  RawConv  raw window -> Conv1d(20->8, k3) -> Conv1d(8->8, k3) over time, flattened,
           concatenated with the features -> 64 -> 32 -> 2                        (~8k params)

forward(x_features, x_raw, sym) with x_raw shaped (B, K, 4L).
"""
import torch
import torch.nn as nn

K, L = 10, 5
RAW_DIM = K * 4 * L


class RawMLP(nn.Module):
    def __init__(self, n_feat, n_sym, emb=4, hidden=(64, 32), n_out=2):
        super().__init__()
        self.emb = nn.Embedding(n_sym, emb)
        layers, d = [], n_feat + emb + RAW_DIM
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, n_out))

    def forward(self, x, raw, s):
        return self.net(torch.cat([x, raw.flatten(1), self.emb(s)], 1))


class RawConv(nn.Module):
    def __init__(self, n_feat, n_sym, emb=4, ch=8, hidden=(64, 32), n_out=2):
        super().__init__()
        self.emb = nn.Embedding(n_sym, emb)
        self.conv = nn.Sequential(nn.Conv1d(4 * L, ch, 3), nn.ReLU(), nn.Conv1d(ch, ch, 3), nn.ReLU())
        layers, d = [], n_feat + emb + ch * (K - 4)
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, n_out))

    def forward(self, x, raw, s):
        z = self.conv(raw.transpose(1, 2)).flatten(1)    # (B, 4L, K) -> (B, ch, K-4) -> flat
        return self.net(torch.cat([x, z, self.emb(s)], 1))


STUDENTS = {"raw_mlp": RawMLP, "raw_conv": RawConv}
