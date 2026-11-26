"""The rep end model. Shapes follow the model contract in docs/data_format.md."""

import torch
from torch import nn

WINDOW = 60    # frames per input window
FEATURES = 12  # x, y for 6 landmarks, after preprocessing


class RepNet(nn.Module):
    """Small 1D CNN: [batch, 60, 12] -> [batch, 60] probability of a rep ending per frame."""

    def __init__(self, channels: int = 32):
        super().__init__()
        layers, in_ch = [], FEATURES
        for dilation in (1, 2, 4):
            layers += [nn.Conv1d(in_ch, channels, 5, padding=2 * dilation, dilation=dilation), nn.ReLU()]
            in_ch = channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels, 1, 1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Use this for training with BCEWithLogitsLoss."""
        return self.head(self.body(x.transpose(1, 2))).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x))
