import torch
import torch.nn as nn
from models.base_model import BaseModel

class StackedGRU(BaseModel):
    def __init__(self, n_features, n_hiddens=100, n_layers=3, dropout=0.0):
        super().__init__()
        self.n_features = n_features
        self.n_hiddens = n_hiddens
        self.n_layers = n_layers
        
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=n_hiddens,
            num_layers=n_layers,
            dropout=dropout,  # dropout 적용
            bidirectional=True,
        )

        self.fc = nn.Linear(n_hiddens * 2, n_features)

    def forward(self, x):
        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(outs[-1])
        return x[0] + out 