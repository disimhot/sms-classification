import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "deepvk/RuModernBERT-base"


class BertClassifier(nn.Module):
    """
    RuModernBERT-based classifier for SMS classification.

    Args:
        num_classes: Number of output classes
        dropout: Dropout probability (default 0.1)
        freeze_bert: Whether to freeze BERT weights (default False)
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.1,
        freeze_bert: bool = False,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.num_classes = num_classes

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            logits: Output logits [batch_size, num_classes]
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]

        return self.classifier(pooled_output)


class BertTokenizerWrapper:
    """
    Tokenizer wrapper for RuModernBERT.

    Args:
        max_length: Maximum sequence length (default 256)
    """

    def __init__(self, max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.max_length = max_length

    def __call__(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """
        Tokenize texts.

        Args:
            texts: List of input texts

        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


class BertDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for BERT inputs.

    Args:
        texts: List of input texts
        labels: Optional list of labels
        tokenizer: BertTokenizerWrapper instance
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None = None,
        tokenizer: BertTokenizerWrapper | None = None,
    ):
        self.texts = texts
        self.labels = labels

        if tokenizer is None:
            tokenizer = BertTokenizerWrapper()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer([self.texts[idx]])

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def collate_fn(batch):
    """Collate function for BERT inputs."""
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=0
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
    )

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    if "labels" in batch[0]:
        labels = torch.stack([item["labels"] for item in batch])
        return inputs, labels

    return inputs
