import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as pl
import pandas as pd
from classification.data.label_encoder import LabelEncoderWrapper
from torch.utils.data import DataLoader, Dataset


@dataclass
class DataModuleConfig:
    """Configuration for SMSDataModule."""

    data_dir: str | Path
    text_column: str = "text"
    label_column: str = "label"
    batch_size: int = 32
    num_workers: int = 0
    train_file: str = "train.csv"
    val_file: str = "val.csv"
    test_file: str = "test.csv"
    predict_file: str = "predict.csv"


@dataclass
class DataManagerConfig:
    """Configuration for SMSDataManager."""

    data_dir: Path
    text_column: str = "text"
    label_column: str = "label"
    train_file: str = "train.csv"
    val_file: str = "val.csv"
    test_file: str = "test.csv"
    predict_file: str = "predict.csv"


@dataclass
class DataFrames:
    """Container for DataFrames."""

    train: pd.DataFrame | None = None
    val: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    predict: pd.DataFrame | None = None


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


class SMSDataModule(pl.LightningDataModule):
    """Lightning DataModule for SMS data."""

    def __init__(
        self,
        config: DataModuleConfig,
        dataframes: DataFrames,
        preprocessor: Any | None = None,
    ):
        super().__init__()

        self.config = config
        self.dataframes = dataframes
        self.preprocessor = preprocessor

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str | None = None):
        """Setup datasets based on stage."""

        if stage == "fit":
            self.train_dataset = SMSDataset(
                texts=self.dataframes.train[self.config.text_column].tolist(),
                labels=self.dataframes.train[self.config.label_column].tolist(),
                preprocessor=self.preprocessor,
            )

        if stage == "validate":
            self.val_dataset = SMSDataset(
                texts=self.dataframes.val[self.config.text_column].tolist(),
                labels=self.dataframes.val[self.config.label_column].tolist(),
                preprocessor=self.preprocessor,
            )

        if stage == "test":
            self.test_dataset = SMSDataset(
                texts=self.dataframes.test[self.config.text_column].tolist(),
                labels=self.dataframes.test[self.config.label_column].tolist(),
                preprocessor=self.preprocessor,
            )

        if stage == "predict":
            self.predict_dataset = SMSDataset(
                texts=self.dataframes.predict[self.config.text_column].tolist(),
                labels=None,
                preprocessor=self.preprocessor,
            )

    def train_dataloader(self) -> DataLoader:
        """Returns DataLoader for training."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns DataLoader for validation."""
        if self.val_dataset is None:
            raise ValueError(
                "Validation dataset is not initialized. "
                "Call setup('fit') or setup('validate') first."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns DataLoader for testing."""
        if self.test_dataset is None:
            raise ValueError("Test dataset is not initialized. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        """Returns DataLoader for prediction."""
        if self.predict_dataset is None:
            raise ValueError("Predict dataset is not initialized. Call setup('predict') first.")
        return DataLoader(
            self.predict_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )


class SMSDataManager:
    """Manager for SMS data."""

    def __init__(self, config: DataManagerConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.label_encoder = LabelEncoderWrapper()
        self._data_cache: dict[str, pd.DataFrame | None] = {
            "train": None,
            "val": None,
            "test": None,
            "predict": None,
        }
        self.id2label: dict[int, str] = {}
        self.label2id: dict[str, int] = {}

    def _read_csv_file(self, filename: str) -> pd.DataFrame | None:
        """Read a CSV file and return a DataFrame."""
        file_path = self.data_dir / filename

        if file_path.exists():
            return pd.read_csv(file_path)
        return None

    def load_all(self) -> dict[str, pd.DataFrame | None]:
        """Load all data csv files and return a dictionary of DataFrames."""
        self._data_cache["train"] = self._read_csv_file(self.config.train_file)
        self._data_cache["val"] = self._read_csv_file(self.config.val_file)
        self._data_cache["test"] = self._read_csv_file(self.config.test_file)
        self._data_cache["predict"] = self._read_csv_file(self.config.predict_file)

        train_df = self._data_cache["train"]
        if train_df is None:
            raise ValueError("train.csv not found — cannot fit LabelEncoder")
        train_labels = train_df["result"]
        encoded = self.label_encoder.fit_transform(train_labels)

        train_df["label"] = encoded
        self._data_cache["train"] = train_df

        self.id2label = dict(enumerate(self.label_encoder.classes_))
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
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        preprocessor: Any | None = None,
    ) -> SMSDataModule:
        """Creates and returns a DataModule."""
        config = DataModuleConfig(
            data_dir=self.data_dir,
            text_column=self.config.text_column,
            label_column=self.config.label_column,
            batch_size=batch_size,
            num_workers=num_workers,
            train_file=self.config.train_file,
            val_file=self.config.val_file,
            test_file=self.config.test_file,
            predict_file=self.config.predict_file,
        )

        dataframes = DataFrames(
            train=self._data_cache["train"],
            val=self._data_cache["val"],
            test=self._data_cache["test"],
            predict=self._data_cache["predict"],
        )

        return SMSDataModule(
            config=config,
            dataframes=dataframes,
            preprocessor=preprocessor,
        )

    def save_label_encoder(self, path: Path | str | None = None) -> Path:
        """
        Save label encoder to JSON file.

        Args:
            path: Path to save file. If None, saves to data_dir/label_encoder.json

        Returns:
            Path to saved file
        """
        path = self.data_dir / "label_encoder.json" if path is None else Path(path)

        encoder_data = {
            "id2label": self.id2label,
            "label2id": self.label2id,
            "classes": list(self.label_encoder.classes_),
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(encoder_data, f, ensure_ascii=False, indent=2)

        return path
