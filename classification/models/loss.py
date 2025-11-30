from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.

    Args:
        labels: List of class labels
        num_classes: Total number of classes

    Returns:
        Tensor of weights for each class
    """
    counts = Counter(labels)
    total = len(labels)

    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reduces loss for well-classified examples, focusing on hard negatives.

    Args:
        alpha: Class weights tensor (optional)
        gamma: Focusing parameter (default 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Computed focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Args:
        num_classes: Number of target classes
        smoothing: Label smoothing factor (default 0.1)
        weight: Class weights tensor (optional)
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Computed loss with label smoothing
        """
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)

        one_hot = torch.full_like(inputs, smooth_value)
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)

        log_probs = F.log_softmax(inputs, dim=1)

        if self.weight is not None:
            weight = self.weight.to(inputs.device)
            log_probs = log_probs * weight.unsqueeze(0)

        loss = -torch.sum(one_hot * log_probs, dim=1)
        return loss.mean()
