import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, padding=3, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class CNN1D(nn.Module):
    def __init__(self, n_leads=12, base_channels=64, n_classes=1, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.conv1 = ConvBlock1D(base_channels, base_channels, kernel_size=7, stride=1, padding=3, dropout=dropout)
        self.conv2 = ConvBlock1D(base_channels, base_channels*2, kernel_size=7, stride=2, padding=3, dropout=dropout)
        self.conv3 = ConvBlock1D(base_channels*2, base_channels*4, kernel_size=7, stride=2, padding=3, dropout=dropout)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(base_channels*4, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        return self.head(x)