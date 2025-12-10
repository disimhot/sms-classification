import lightning as L
import torch
import torchmetrics
from classification.models.loss import FocalLoss
from torch import nn


class SMSClassificationModule(L.LightningModule):
    """
    Lightning module for multi-class SMS classification.

    Args:
        model: Neural network model (BERT or MLP)
        num_classes: Number of target classes (default 13)
        class_weights: Optional class weights for imbalanced data
        learning_rate: Learning rate for optimizer
        loss_type: Loss function type ('cross_entropy', 'focal', 'nll')
        focal_gamma: Gamma parameter for Focal Los
        scheduler_eta_min: Minimum learning rate for scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 13,
        class_weights: torch.Tensor | None = None,
        learning_rate: float = 1e-3,
        loss_type: str = "cross_entropy",
        focal_gamma: float = 2.0,
        scheduler_eta_min: float = 1e-7,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "criterion"])

        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.scheduler_eta_min = scheduler_eta_min

        # Loss
        self.criterion = self._create_loss(loss_type, class_weights, focal_gamma)

        # Validation metrics
        self.val_f1_weighted = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.val_f1_macro = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_auroc_weighted = torchmetrics.AUROC(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.val_auroc_macro = torchmetrics.AUROC(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # Test metrics
        self.test_f1_weighted = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.test_f1_macro = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_auroc_weighted = torchmetrics.AUROC(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.test_auroc_macro = torchmetrics.AUROC(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def _create_loss(self, loss_type, class_weights, focal_gamma):
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(weight=class_weights)
        if loss_type == "focal":
            return FocalLoss(alpha=class_weights, gamma=focal_gamma)
        if loss_type == "nll":
            return nn.NLLLoss(weight=class_weights)
        raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, inputs) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            inputs: Input data (dict with input_ids, attention_mask for BERT;
                   tensor of embeddings for MLP)

        Returns:
            logits: Model output logits [batch_size, num_classes]
        """
        return self.model(inputs)

    def training_step(self, batch) -> torch.Tensor:
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.val_accuracy(preds, targets)
        self.val_f1_weighted(preds, targets)
        self.val_f1_macro(preds, targets)
        self.val_auroc_weighted(probs, targets)
        self.val_auroc_macro(probs, targets)

        self.log(
            "val_accuracy",
            self.val_accuracy,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_f1_weighted",
            self.val_f1_weighted,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_f1_macro",
            self.val_f1_macro,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_auroc_weighted",
            self.val_auroc_weighted,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_auroc_macro",
            self.val_auroc_macro,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch):
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        self.log("test_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.test_accuracy(preds, targets)
        self.test_f1_weighted(preds, targets)
        self.test_f1_macro(preds, targets)
        self.test_auroc_weighted(probs, targets)
        self.test_auroc_macro(probs, targets)

        self.log(
            "test_accuracy",
            self.test_accuracy,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_f1_weighted",
            self.test_f1_weighted,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_f1_macro",
            self.test_f1_macro,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_auroc_weighted",
            self.test_auroc_weighted,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_auroc_macro",
            self.test_auroc_macro,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def predict_step(self, batch):
        """Generate predictions for inference."""
        if isinstance(batch, tuple):
            inputs, _ = batch
        else:
            inputs = batch

        logits = self.forward(inputs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        return {"predictions": preds, "probabilities": probs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.scheduler_eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
