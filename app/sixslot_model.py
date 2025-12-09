import torch
import torch.nn as nn

class PokemonEncoder(nn.Module):
    def __init__(self, in_dim=18, hidden=256, mlp_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(mlp_layers):
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            dim = hidden
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        B, S, D = x.shape
        x = x.reshape(B*S, D)
        out = self.mlp(x)
        return out.reshape(B, S, -1)


class TeamPredictor(nn.Module):
    def __init__(self, hidden=256, num_classes=1000, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


class SixSlotModel(nn.Module):
    def __init__(self, in_dim=18, hidden=256, num_classes=1000):
        super().__init__()
        self.encoder = PokemonEncoder(in_dim=in_dim, hidden=hidden)
        self.predictor = TeamPredictor(hidden=hidden, num_classes=num_classes)

    def forward(self, x, winrate=None):
        enc = self.encoder(x)   # (B,5,H)
        pooled = enc.sum(dim=1)
        return self.predictor(pooled)