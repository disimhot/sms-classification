from pathlib import Path

import numpy as np
import torch
from gensim.models import Word2Vec


class Word2VecEmbedder:
    """
    Word2Vec embedder for converting texts to vectors.

    Args:
        vector_size: Dimension of word vectors (default 300)
        window: Context window size (default 5)
        min_count: Minimum word frequency (default 1)
        workers: Number of workers for training (default 4)
        sg: Training algorithm (0=CBOW, 1=Skip-gram, default 1)
    """

    def __init__(
        self,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        sg: int = 1,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model: Word2Vec | None = None

    def fit(self, tokenized_texts: list[list[str]], epochs: int = 10):
        """Train Word2Vec model on tokenized texts.

        Args:
            tokenized_texts: List of tokenized sentences
            epochs: Number of training epochs

        Returns:
            Self for method chaining
        """
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=epochs,
        )
        return self

    def transform(self, tokenized_texts: list[list[str]]) -> np.ndarray:
        """Transform tokenized texts to embeddings using mean pooling.

        Args:
            tokenized_texts: List of tokenized sentences

        Returns:
            Array of embeddings [num_texts, vector_size]
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        embeddings = []
        for tokens in tokenized_texts:
            word_vectors = []
            for token in tokens:
                if token in self.model.wv:
                    word_vectors.append(self.model.wv[token])

            if word_vectors:
                embedding = np.mean(word_vectors, axis=0)
            else:
                embedding = np.zeros(self.vector_size)

            embeddings.append(embedding)

        return np.array(embeddings)

    def fit_transform(
        self,
        tokenized_texts: list[list[str]],
        epochs: int = 10,
    ) -> np.ndarray:
        """Fit model and transform texts in one step.

        Args:
            tokenized_texts: List of tokenized sentences
            epochs: Number of training epochs

        Returns:
            Array of embeddings [num_texts, vector_size]
        """
        self.fit(tokenized_texts, epochs)
        return self.transform(tokenized_texts)

    def save(self, path: str | Path) -> None:
        """Save Word2Vec model to file."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        self.model.save(str(path))

    def load(self, path: str | Path):
        """
        Load Word2Vec model from file.

        Args:
            path: Path to saved model

        Returns:
            Self for method chaining
        """
        self.model = Word2Vec.load(str(path))
        self.vector_size = self.model.vector_size
        return self


class Word2VecDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for Word2Vec embeddings.

    Args:
        embeddings: Numpy array of embeddings
        labels: Optional list of labels
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: list[int] | None = None,
    ):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = labels
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.embeddings[idx]
        return self.embeddings[idx], self.labels[idx]
