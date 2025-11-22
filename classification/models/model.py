from transformers import AutoModelForSequenceClassification


def load_model(
    pretrained_name: str, num_labels: int, id2label=None, label2id=None
) -> AutoModelForSequenceClassification:
    """Loads model from HuggingFace."""
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
