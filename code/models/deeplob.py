"""DeepLOB (Zhang, Zohren & Roberts 2019, arXiv 1808.03668), following the authors' PyTorch reference.

Input: (batch, 1, T=100, 40) -- the last T snapshots of a 10-level book, columns per level
(ask price, ask size, bid price, bid size) as in FI-2010. Output: logits over n_out classes.

Blocks: three conv blocks that first pair price/size, then pair bid/ask, then merge all 10 levels;
an inception module (1x1 -> 3x1, 1x1 -> 5x1, maxpool -> 1x1) giving 192 channels; an LSTM(64) over time.
~144k parameters, ~70M FLOPs per prediction at T=100.
"""
import torch
import torch.nn as nn


def _bn(c):
    return nn.BatchNorm2d(c)


class DeepLOB(nn.Module):
    def __init__(self, n_out=3, lstm_hidden=64):
        super().__init__()
        lrelu = lambda: nn.LeakyReLU(0.01)  # noqa: E731
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 2), stride=(1, 2)), lrelu(), _bn(32),
            nn.Conv2d(32, 32, (4, 1)), lrelu(), _bn(32),
            nn.Conv2d(32, 32, (4, 1)), lrelu(), _bn(32))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 2), stride=(1, 2)), nn.Tanh(), _bn(32),
            nn.Conv2d(32, 32, (4, 1)), nn.Tanh(), _bn(32),
            nn.Conv2d(32, 32, (4, 1)), nn.Tanh(), _bn(32))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 10)), lrelu(), _bn(32),
            nn.Conv2d(32, 32, (4, 1)), lrelu(), _bn(32),
            nn.Conv2d(32, 32, (4, 1)), lrelu(), _bn(32))
        self.inp1 = nn.Sequential(nn.Conv2d(32, 64, 1), lrelu(), _bn(64),
                                  nn.Conv2d(64, 64, (3, 1), padding="same"), lrelu(), _bn(64))
        self.inp2 = nn.Sequential(nn.Conv2d(32, 64, 1), lrelu(), _bn(64),
                                  nn.Conv2d(64, 64, (5, 1), padding="same"), lrelu(), _bn(64))
        self.inp3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)),
                                  nn.Conv2d(32, 64, 1), lrelu(), _bn(64))
        self.lstm = nn.LSTM(192, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_out)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))                     # (B, 32, T-12, 1)
        x = torch.cat([self.inp1(x), self.inp2(x), self.inp3(x)], 1)  # (B, 192, T-12, 1)
        x = x.permute(0, 2, 1, 3).flatten(2)                          # (B, T-12, 192)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])
