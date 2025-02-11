import torch
import torch.nn as nn

class StackedLSTM(nn.Module):
    def __init__(self, n_features, n_hiddens=200, n_layers=3, dropout=0.1, bidirectional=True):
        super().__init__()
        self.n_features = n_features
        self.n_hiddens = n_hiddens
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        # LSTM 레이어
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hiddens,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 양방향이면 hidden_size * 2
        lstm_output_size = n_hiddens * 2 if bidirectional else n_hiddens
        
        # 출력 레이어들
        self.fc = nn.Linear(lstm_output_size, n_features)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.sigmoid = nn.Sigmoid()
        self.dense1 = nn.Linear(n_features, n_features // 2)
        self.dense2 = nn.Linear(n_features // 2, n_features)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        
        # LSTM 통과
        lstm_out, _ = self.rnn(x)
        
        # 마지막 시퀀스의 출력만 사용
        last_output = lstm_out[:, -1, :]
        
        # FC 레이어 통과
        out = self.fc(last_output)
        out = self.relu(out)
        
        # Dense 레이어들 통과
        out = self.dense1(out)
        out = self.relu(out)
        out = self.dense2(out)
        
        return out 