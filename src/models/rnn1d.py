import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN1D(nn.Module):
    def __init__(self, n_leads=12, hidden_size=64, num_layers=2, dropout=0.2, n_classes=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=n_leads,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x shape: [batch, n_leads, n_samples]
        x = x.transpose(1, 2)  # [batch, n_samples, n_leads]
        rnn_out, _ = self.rnn(x)  # [batch, n_samples, hidden_size]
        rnn_out = rnn_out.transpose(1, 2)  # [batch, hidden_size, n_samples]
        x = self.gap(rnn_out)  # [batch, hidden_size, 1]
        x = x.squeeze(-1)  # [batch, hidden_size]
        return self.head(x)