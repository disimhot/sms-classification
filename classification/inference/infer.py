import sys
from pathlib import Path

import lightning as L
import torch
from classification.data.dataset_loader import SMSDataManager
from classification.data.preprocess import TextPreprocessor
from classification.models.bert import BertClassifier, BertDataset, BertTokenizerWrapper, collate_fn
from classification.models.mlp import MLPClassifier
from classification.models.word2vec import Word2VecDataset, Word2VecEmbedder
from classification.module.module import SMSClassificationModule
from classification.utils.commit_id import _get_git_commit
from classification.utils.config import load_config
from classification.utils.data_util import ensure_data_downloaded


def infer(overrides: list[str] | None = None):
    """
    Run inference on test dataset.

    Args:
        overrides: List of Hydra config overrides, e.g. ["models=mlp"]
    """
    cfg = load_config(overrides)

    data_dir = Path(cfg.data.data_dir)
    required_paths = [str(data_dir / cfg.data.test_file)]

    if not ensure_data_downloaded(required_paths):
        print("Failed to download required data")
        sys.exit()

    # Check model exists
    model_path = Path(cfg.models.output_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Please run training first: python commands.py train")
        sys.exit()

    # Load data
    manager = SMSDataManager(data_dir)
    data = manager.load_all()

    test_texts = data["test"]["text"].tolist()
    test_labels = data["test"]["label"].tolist()

    # Create model and dataloader based on model type
    if cfg.models.type == "mlp":
        model, test_loader = _setup_mlp(cfg, test_texts, test_labels)
    elif cfg.models.type == "bert":
        model, test_loader = _setup_bert(cfg, test_texts, test_labels)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Create Lightning module
    module = SMSClassificationModule(
        model=model,
        num_classes=cfg.module.num_classes,
        learning_rate=cfg.module.optimizer.learning_rate,
        loss_type=cfg.module.loss.type,
        focal_gamma=cfg.module.loss.focal_gamma,
        scheduler_eta_min=cfg.module.scheduler.eta_min,
    )

    mlflow_logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
        save_dir=cfg.logging.save_dir,
    )
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "stage", "inference")
    mlflow_logger.experiment.log_param(mlflow_logger.run_id, "git_commit", _get_git_commit())

    # Trainer for testing
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=mlflow_logger,
    )

    # Run test
    results = trainer.test(module, dataloaders=test_loader)

    print("\n=== Test Results ===")
    for key, value in results[0].items():
        print(f"{key}: {value:.4f}")

    return results


def _setup_mlp(cfg, test_texts, test_labels):
    """Setup MLP model for inference."""
    # Check Word2Vec model exists
    w2v_path = Path(cfg.models.word2vec.output_path)
    if not w2v_path.exists():
        print(f"Word2Vec model not found: {w2v_path}")
        print("Please run training first: python commands.py train models=mlp")
        sys.exit()

    preprocessor = TextPreprocessor(language="russian")
    test_clean = [preprocessor.preprocess_text(t) for t in test_texts]
    test_tokenized = [text.split() for text in test_clean]

    # Load Word2Vec
    embedder = Word2VecEmbedder()
    embedder.load(w2v_path)
    test_embeddings = embedder.transform(test_tokenized)

    # Dataset
    test_dataset = Word2VecDataset(test_embeddings, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )

    model = MLPClassifier(
        input_dim=cfg.models.word2vec.vector_size,
        num_classes=cfg.module.num_classes,
        dropout=cfg.models.dropout,
    )

    return model, test_loader


def _setup_bert(cfg, test_texts, test_labels):
    """Setup BERT model for inference."""
    tokenizer = BertTokenizerWrapper(
        pretrained_model=cfg.models.pretrained_model,
        max_length=cfg.models.max_length,
    )

    test_dataset = BertDataset(test_texts, test_labels, tokenizer=tokenizer)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = BertClassifier(
        num_classes=cfg.module.num_classes,
        pretrained_model=cfg.models.pretrained_model,
        dropout=cfg.models.dropout,
        freeze_bert=cfg.models.freeze_bert,
    )

    return model, test_loader
