import pandas as pd
from sklearn.preprocessing import LabelEncoder


class LabelEncoderWrapper:
    """Обертка над LabelEncoder для сохранения id2label и label2id."""

    def __init__(self):
        self.encoder = LabelEncoder()
        self.id2label = None
        self.label2id = None
        self.classes_ = None

    def fit(self, labels: pd.Series):
        """Обучает энкодер на колонке меток."""
        self.encoder.fit(labels)
        self.id2label = {i: label for i, label in enumerate(self.encoder.classes_)}
        self.label2id = {label: i for i, label in self.id2label.items()}
        self.classes_ = self.encoder.classes_

    def transform(self, labels):
        """Преобразует в цифровые метки."""
        return self.encoder.transform(labels)

    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)

    def inverse_transform(self, ids):
        """Переводит числовые метки обратно в текст."""
        return self.encoder.inverse_transform(ids)
