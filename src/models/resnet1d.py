"""
1D ResNet for 12-lead ECG classification (Brugada vs Normal).

Architecture
------------
  Input  : (B, 12, 1200)
  Stem   : Conv1d(12→64, k=15, s=2) + BN + ReLU + MaxPool(s=2)  →  (B, 64, 300)
  Layer1 : 2 × ResBlock(64,   64,  k=7, s=1)  →  (B,  64, 300)
  Layer2 : 2 × ResBlock(64,  128,  k=7, s=2)  →  (B, 128, 150)
  Layer3 : 2 × ResBlock(128, 256,  k=7, s=2)  →  (B, 256,  75)
  Layer4 : 2 × ResBlock(256, 512,  k=7, s=2)  →  (B, 512,  38)
  GAP    : AdaptiveAvgPool1d(1)                →  (B, 512)
  Head   : Dropout → Linear(512, 1)           →  (B, 1)  logit

Use with BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """
    Basic 1-D residual block.

      Conv → BN → ReLU → Dropout → Conv → BN
      + skip (1×1 conv when channels or stride change)
      → ReLU
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class ResNet1D(nn.Module):
    """
    1D ResNet for multi-lead ECG binary classification.

    Parameters
    ----------
    n_leads       : input channels (12 for a standard 12-lead ECG)
    base_channels : width of the first residual stage (doubled each stage)
    kernel_size   : temporal kernel size for all residual blocks
    dropout       : dropout probability inside blocks and in the head
    """

    def __init__(
        self,
        n_leads: int = 12,
        base_channels: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        C = base_channels

        # Stem: two successive halving steps  1200 → 600 → 300
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, C, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(C,   C,   n=2, stride=1, ks=kernel_size, dp=dropout)
        self.layer2 = self._make_layer(C,   C*2, n=2, stride=2, ks=kernel_size, dp=dropout)
        self.layer3 = self._make_layer(C*2, C*4, n=2, stride=2, ks=kernel_size, dp=dropout)
        self.layer4 = self._make_layer(C*4, C*8, n=2, stride=2, ks=kernel_size, dp=dropout)
        # 300 → 300 → 150 → 75 → 38

        self.gap      = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(dropout)
        self.fc        = nn.Linear(C * 8, 1)

        self._init_weights()

    @staticmethod
    def _make_layer(in_ch, out_ch, n, stride, ks, dp) -> nn.Sequential:
        layers = [ResidualBlock1D(in_ch, out_ch, ks, stride, dp)]
        for _ in range(1, n):
            layers.append(ResidualBlock1D(out_ch, out_ch, ks, 1, dp))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 12, 1200)

        Returns
        -------
        logit : (B, 1) — pass through sigmoid for probability
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)    # (B, C*8)
        x = self.head_drop(x)
        return self.fc(x)              # (B, 1)

    def get_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (layer4_output, logit) for GradCAM.

        layer4_output : (B, C*8, L_last)
        logit         : (B, 1)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)

        out = self.gap(feat).squeeze(-1)
        out = self.head_drop(out)
        logit = self.fc(out)
        return feat, logit

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
