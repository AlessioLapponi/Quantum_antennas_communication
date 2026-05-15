import torch
import torch.nn as nn


class CurveCapacityMLP(nn.Module):
    """
    Curve-output neural surrogate.

    Input:
        physical feature vector

    Output:
        full C_E(t) curve on the fixed training time grid.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=256,
        num_hidden_layers=5,
        dropout=0.0,
        use_softplus=True,
        causal_mask=None,
    ):
        super().__init__()

        self.use_softplus = use_softplus

        layers = []
        current_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)

        if causal_mask is not None:
            causal_mask = torch.as_tensor(causal_mask, dtype=torch.float32)
            self.register_buffer("causal_mask", causal_mask.view(1, -1))
        else:
            self.causal_mask = None

        self.softplus = nn.Softplus()

    def forward(self, x):
        raw = self.net(x)

        if self.use_softplus:
            y = self.softplus(raw)
        else:
            y = raw

        if self.causal_mask is not None:
            y = y * self.causal_mask

        return y