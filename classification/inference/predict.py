import json
from pathlib import Path

import torch
from classification.data.preprocess import TextPreprocessor
from classification.models.bert import BertClassifier, BertTokenizerWrapper
from classification.models.mlp import MLPClassifier
from classification.models.word2vec import Word2VecEmbedder
from classification.utils.config import load_config


class ModelNotFoundError(Exception):
    """Raised when model weights file is not found."""


class LabelEncoderNotFoundError(Exception):
    """Raised when label encoder file is not found."""


class Predictor:
    """
    Predictor for SMS classification.

    Loads model weights and provides prediction interface.

    Args:
        model_type: Type of model ('bert' or 'mlp')
    """

    def __init__(self, model_type: str = "bert"):
        self.model_type = model_type
        self.cfg = load_config(overrides=[f"models={model_type}"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.preprocessor = None
        self.embedder = None
        self.id2label: dict[int, str] = {}
        self.label2id: dict[str, int] = {}

    def load(self) -> None:
        """Load model and label encoder."""
        self._load_label_encoder()
        self._load_model()

    def _load_label_encoder(self) -> None:
        """Load label encoder from JSON file."""
        encoder_path = Path(self.cfg.data.data_dir) / "label_encoder.json"

        if not encoder_path.exists():
            raise LabelEncoderNotFoundError(
                f"Label encoder not found: {encoder_path}. "
                "Please run training first to generate label_encoder.json"
            )

        with encoder_path.open(encoding="utf-8") as f:
            data = json.load(f)

        self.id2label = {int(k): v for k, v in data["id2label"].items()}
        self.label2id = data["label2id"]

    def _load_model(self) -> None:
        """Load model weights."""
        model_path = Path(self.cfg.models.output_path)

        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model not found: {model_path}. "
                f"Please train the model first: python commands.py train models={self.model_type}"
            )

        if self.model_type == "bert":
            self._load_bert(model_path)
        elif self.model_type == "mlp":
            self._load_mlp(model_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self._validate_num_classes()

    def _validate_num_classes(self) -> None:
        """Validate that model num_classes matches label_encoder."""
        if self.model_type == "bert":
            model_num_classes = self.model.classifier[1].out_features
        else:
            model_num_classes = self.model.model[6].out_features

        encoder_num_classes = len(self.id2label)

        if model_num_classes != encoder_num_classes:
            raise RuntimeError(
                f"Mismatch: model has {model_num_classes} classes, "
                f"but label_encoder has {encoder_num_classes}. "
                "Please retrain the model or update label_encoder.json"
            )

    def _load_bert(self, model_path: Path) -> None:
        """Load BERT model and tokenizer."""
        self.tokenizer = BertTokenizerWrapper(
            pretrained_model=self.cfg.models.pretrained_model,
            max_length=self.cfg.models.max_length,
        )

        # Определяем num_classes из сохранённых весов
        state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
        num_classes = state_dict["classifier.1.weight"].shape[0]

        self.model = BertClassifier(
            num_classes=num_classes,
            pretrained_model=self.cfg.models.pretrained_model,
            dropout=self.cfg.models.dropout,
            freeze_bert=self.cfg.models.freeze_bert,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _load_mlp(self, model_path: Path) -> None:
        """Load MLP model, Word2Vec embedder, and preprocessor."""
        w2v_path = Path(self.cfg.models.word2vec.output_path)

        if not w2v_path.exists():
            raise ModelNotFoundError(
                f"Word2Vec model not found: {w2v_path}. "
                "Please train the MLP model first: python commands.py train models=mlp"
            )

        self.preprocessor = TextPreprocessor(language="russian")

        self.embedder = Word2VecEmbedder()
        self.embedder.load(w2v_path)

        state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
        num_classes = state_dict["model.6.weight"].shape[0]

        self.model = MLPClassifier(
            input_dim=self.cfg.models.word2vec.vector_size,
            num_classes=num_classes,
            dropout=self.cfg.models.dropout,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[dict]:
        """
        Predict classes for input texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of prediction dictionaries with keys:
                - text: Original input text
                - label: Predicted class label
                - label_id: Predicted class ID
                - confidence: Prediction confidence
                - probabilities: Dict of all class probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self.model_type == "bert":
            inputs = self._prepare_bert_inputs(texts)
        else:
            inputs = self._prepare_mlp_inputs(texts)

        logits = self.model(inputs)
        probs = torch.softmax(logits, dim=1)
        confidences, pred_ids = torch.max(probs, dim=1)

        results = []
        for i, text in enumerate(texts):
            pred_id = pred_ids[i].item()

            probabilities = {
                self.id2label[j]: round(probs[i, j].item(), 4) for j in range(len(self.id2label))
            }

            results.append(
                {
                    "text": text,
                    "label": self.id2label[pred_id],
                    "label_id": pred_id,
                    "confidence": round(confidences[i].item(), 4),
                    "probabilities": probabilities,
                }
            )

        return results

    def _prepare_bert_inputs(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize texts for BERT."""
        encoded = self.tokenizer(texts)
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _prepare_mlp_inputs(self, texts: list[str]) -> torch.Tensor:
        """Preprocess and embed texts for MLP."""
        clean_texts = [self.preprocessor.preprocess_text(t) for t in texts]
        tokenized = [text.split() for text in clean_texts]
        embeddings = self.embedder.transform(tokenized)
        return torch.tensor(embeddings, dtype=torch.float32).to(self.device)

    def get_model_path(self) -> str:
        """Return path to model weights."""
        return str(self.cfg.models.output_path)
