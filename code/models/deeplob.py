"""DeepLOB (Zhang, Zohren & Roberts 2019, arXiv 1808.03668).

Input: (batch, 1, T=100, 40) -- the last T snapshots of a 10-level book, columns per level
(ask price, ask size, bid price, bid size) as in FI-2010 (paper eq. 6). Output: logits over n_out classes.

Default sizes follow the paper (Fig. 3-4, Sec. IV): conv blocks with 16 filters, Inception@32, LSTM(64),
~60k parameters, zero padding on every conv so the time dimension stays T ("We apply zero padding to every
convolutional layer so the time dimension of our inputs does not change"). Activations follow the authors'
released code: Leaky-ReLU(0.01), except tanh in the second conv block.

variant="repo" gives the larger version in the authors' later PyTorch notebook (32 filters, Inception@64,
no time padding, ~144k parameters).
"""
import torch
import torch.nn as nn

VARIANTS = {"paper": dict(conv=16, inc=32, pad=True), "repo": dict(conv=32, inc=64, pad=False)}


class DeepLOB(nn.Module):
    def __init__(self, n_out=3, lstm_hidden=64, variant="paper"):
        super().__init__()
        v = VARIANTS[variant]
        c, k, pad = v["conv"], v["inc"], ("same" if v["pad"] else 0)
        lrelu = lambda: nn.LeakyReLU(0.01)  # noqa: E731

        def block(c_in, width_kernel, stride, act):
            return nn.Sequential(
                nn.Conv2d(c_in, c, (1, width_kernel), stride=(1, stride)), act(), nn.BatchNorm2d(c),
                nn.Conv2d(c, c, (4, 1), padding=pad), act(), nn.BatchNorm2d(c),
                nn.Conv2d(c, c, (4, 1), padding=pad), act(), nn.BatchNorm2d(c))

        self.conv1 = block(1, 2, 2, lrelu)        # pairs (price, size) at each level
        self.conv2 = block(c, 2, 2, nn.Tanh)      # pairs ask/bid -> micro-price-like maps
        self.conv3 = block(c, 10, 1, lrelu)       # merges all 10 levels
        self.inp1 = nn.Sequential(nn.Conv2d(c, k, 1), lrelu(), nn.BatchNorm2d(k),
                                  nn.Conv2d(k, k, (3, 1), padding="same"), lrelu(), nn.BatchNorm2d(k))
        self.inp2 = nn.Sequential(nn.Conv2d(c, k, 1), lrelu(), nn.BatchNorm2d(k),
                                  nn.Conv2d(k, k, (5, 1), padding="same"), lrelu(), nn.BatchNorm2d(k))
        self.inp3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)),
                                  nn.Conv2d(c, k, 1), lrelu(), nn.BatchNorm2d(k))
        self.lstm = nn.LSTM(3 * k, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_out)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))                     # (B, c, T', 1)
        x = torch.cat([self.inp1(x), self.inp2(x), self.inp3(x)], 1)  # (B, 3k, T', 1)
        x = x.permute(0, 2, 1, 3).flatten(2)                          # (B, T', 3k)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])
