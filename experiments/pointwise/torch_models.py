import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """
    Adds sinusoidal time features.

    Input x shape:
        [batch, 5]

    Columns expected:
        gamma_mean, gamma_delta, omega_mean, omega_delta, t_scaled

    We keep the original 5 features and append sin/cos features of t_scaled.
    """

    def __init__(self, num_frequencies=8):
        super().__init__()
        self.num_frequencies = num_frequencies

        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)

    def forward(self, x):
        t = x[:, -1:]  # last column is scaled time

        angles = 2.0 * torch.pi * t * self.frequencies.view(1, -1)

        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)

        return torch.cat([x, sin_features, cos_features], dim=1)


class CapacityMLP(nn.Module):
    def __init__(
        self,
        input_dim=5,
        hidden_dim=256,
        num_hidden_layers=4,
        num_frequencies=8,
        dropout=0.0,
    ):
        super().__init__()

        self.fourier = FourierFeatures(num_frequencies=num_frequencies)

        expanded_dim = input_dim + 2 * num_frequencies

        layers = []

        current_dim = expanded_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fourier(x)
        return self.net(x).squeeze(-1)