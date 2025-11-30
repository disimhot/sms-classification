import torch
from torch import nn


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for text classification.

    Designed for use with pre-computed embeddings (e.g., Word2Vec).

    Args:
        input_dim: Dimension of input embeddings
        num_classes: Number of output classes
        dropout: Dropout probability (default 0.3)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input embeddings [batch_size, input_dim]

        Returns:
            logits: Output logits [batch_size, num_classes]
        """
        return self.model(x)
