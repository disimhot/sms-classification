from enum import Enum
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
from classification.data.label_encoder import LabelEncoderWrapper
from torch.utils.data import DataLoader, Dataset


class DataFileType(Enum):
    """Types of available data files."""

    TRAIN = "train.csv"
    VAL = "val.csv"
    TEST = "test.csv"
    PREDICT = "predict.csv"


class SMSDataset(Dataset):
    """Dataset for SMS data."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None = None,
        preprocessor: Any | None = None,
    ):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]

        if self.preprocessor is not None:
            text = self.preprocessor.preprocess_text(text)

        if self.labels is None:
            return text
        return text, self.labels[idx]


class SMSDataModule(L.LightningDataModule):
    """Lightning DataModule for SMS data."""

    def __init__(
        self,
        data_dir: str | Path,
        text_column: str = "text",
        label_column: str = "label",
        batch_size: int = 32,
        num_workers: int = 0,
        preprocessor: Any | None = None,
        train_df: pd.DataFrame | None = None,
        val_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
        predict_df: pd.DataFrame | None = None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.text_column = text_column
        self.label_column = label_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessor = preprocessor

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.predict_df = predict_df

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str | None = None):
        """Setup datasets based on stage."""

        if stage == "fit":
            self.train_dataset = SMSDataset(
                texts=self.train_df[self.text_column].tolist(),
                labels=self.train_df[self.label_column].tolist(),
                preprocessor=self.preprocessor,
            )

        if stage == "validate":
            self.val_dataset = SMSDataset(
                texts=self.val_df[self.text_column].tolist(),
                labels=self.val_df[self.label_column].tolist(),
                preprocessor=self.preprocessor,
            )

        if stage == "test":
            self.test_dataset = SMSDataset(
                texts=self.test_df[self.text_column].tolist(),
                labels=self.test_df[self.label_column].tolist(),
                preprocessor=self.preprocessor,
            )

        if stage == "predict":
            self.predict_dataset = SMSDataset(
                texts=self.predict_df[self.text_column].tolist(),
                labels=None,
                preprocessor=self.preprocessor,
            )

    def train_dataloader(self) -> DataLoader:
        """Returns DataLoader for training."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns DataLoader for validation."""
        if self.val_dataset is None:
            raise ValueError(
                "Validation dataset is not initialized. Call setup('fit') or setup('validate') first."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для тестирования."""
        if self.test_dataset is None:
            raise ValueError("Test dataset is not initialized. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns DataLoader for prediction."""
        if self.predict_dataset is None:
            raise ValueError("Predict dataset is not initialized. Call setup('predict') first.")
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class SMSDataManager:
    """Manager for SMS data."""

    def __init__(self, data_dir: Path, text_column: str = "text", label_column: str = "label"):
        self.data_dir = Path(data_dir)
        self.label_encoder = LabelEncoderWrapper()
        self.text_column = text_column
        self.label_column = label_column
        self._data_cache: dict[str, pd.DataFrame | None] = {
            "train": None,
            "val": None,
            "test": None,
            "predict": None,
        }
        self.id2label = []
        self.label2id = []

    def _read_csv_file(self, file_type: DataFileType) -> pd.DataFrame | None:
        """Read a CSV file and return a DataFrame."""
        file_path = self.data_dir / file_type.value

        if file_path.exists():
            return pd.read_csv(file_path)
        return None

    def load_all(self) -> dict[str, pd.DataFrame | None]:
        """Load all data csv files and return a dictionary of DataFrames."""
        self._data_cache["train"] = self._read_csv_file(DataFileType.TRAIN)
        self._data_cache["val"] = self._read_csv_file(DataFileType.VAL)
        self._data_cache["test"] = self._read_csv_file(DataFileType.TEST)
        self._data_cache["predict"] = self._read_csv_file(DataFileType.PREDICT)

        train_df = self._data_cache["train"]
        if train_df is None:
            raise ValueError("train.csv not found — cannot fit LabelEncoder")
        train_labels = train_df["result"]
        encoded = self.label_encoder.fit_transform(train_labels)

        train_df["label"] = encoded
        self._data_cache["train"] = train_df

        self.id2label = {i: lbl for i, lbl in enumerate(self.label_encoder.classes_)}
        self.label2id = {lbl: i for i, lbl in self.id2label.items()}

        for key in ["val", "test"]:
            df = self._data_cache[key]
            if df is not None:
                if "result" not in df.columns:
                    raise ValueError(f"{key}.csv must contain 'result'")
                df["label"] = self.label_encoder.transform(df["result"])
                self._data_cache[key] = df

        return self._data_cache

    def create_datamodule(
        self, batch_size: int = 32, num_workers: int = 0, preprocessor: Any | None = None
    ) -> SMSDataModule:
        """Creates and returns a DataModule."""

        return SMSDataModule(
            data_dir=self.data_dir,
            text_column=self.text_column,
            label_column=self.label_column,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocessor=preprocessor,
            train_df=self._data_cache["train"],
            val_df=self._data_cache["val"],
            test_df=self._data_cache["test"],
            predict_df=self._data_cache["predict"],
        )
