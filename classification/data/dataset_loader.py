from transformers import AutoTokenizer

def get_tokenizer(model_name: str, max_length: int = 256):
    """
    Загружает токенизатор и возвращает его.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return tokenizer, tokenize_function


def tokenize_datasets(datasets_dict, tokenize_fn):
    """
    Применяет токенизацию к HuggingFace DatasetDict и форматирует под torch.
    """
    datasets_tokenized = datasets_dict.map(tokenize_fn, batched=True, desc="Tokenizing")
    columns = ["input_ids", "attention_mask", "label"]
    datasets_tokenized.set_format(type="torch", columns=columns)
    return datasets_tokenized
