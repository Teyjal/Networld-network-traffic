import torch
import torch.nn as nn


class NetWorldLSTM(nn.Module):

    def __init__(
        self,
        input_size=36,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.infiltration_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_hidden = output[:, -1, :]

        logits = self.infiltration_head(last_hidden)

        return logits