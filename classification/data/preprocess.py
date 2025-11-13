import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def load_and_split_dataset(csv_path: str, text_col: str = "text", label_col: str = "label", test_size: float = 0.2, seed: int = 42):
    """
    Загружает датасет из CSV, делит 80/20 и возвращает DatasetDict для HuggingFace.
    """
    df = pd.read_csv(csv_path)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[label_col])

    train = Dataset.from_pandas(train_df[[text_col, label_col]], preserve_index=False)
    test = Dataset.from_pandas(test_df[[text_col, label_col]], preserve_index=False)

    datasets_dict = DatasetDict({
        "train": train,
        "test": test
    })
    return datasets_dict
