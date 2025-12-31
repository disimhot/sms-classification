import sys
from pathlib import Path

import lightning as pl
import torch
from classification.data.dataset_loader import DataManagerConfig, SMSDataManager
from classification.data.preprocess import TextPreprocessor
from classification.models.bert import (
    BertClassifier,
    BertDataset,
    BertTokenizerWrapper,
    collate_fn,
)
from classification.models.loss import compute_class_weights
from classification.models.mlp import MLPClassifier
from classification.models.word2vec import Word2VecDataset, Word2VecEmbedder
from classification.module.module import SMSClassificationModule
from classification.utils.commit_id import _get_git_commit
from classification.utils.config import load_config
from classification.utils.data_util import ensure_data_downloaded
from omegaconf import OmegaConf


def train(*overrides):
    """
    Train SMS classification model.

    Args:
        overrides: Hydra config overrides, e.g. models=mlp training.max_epochs=5
    """
    cfg = load_config(list(overrides) if overrides else None)

    data_dir = Path(cfg.data.data_dir)
    required_paths = [
        str(data_dir / cfg.data.train_file),
        str(data_dir / cfg.data.val_file),
        str(data_dir / cfg.data.test_file),
    ]

    if not ensure_data_downloaded(required_paths):
        print("Failed to download required data")
        sys.exit()

    pl.seed_everything(cfg.seed)

    data_config = DataManagerConfig(
        data_dir=data_dir,
        train_file=cfg.data.train_file,
        val_file=cfg.data.val_file,
        test_file=cfg.data.test_file,
        predict_file=cfg.data.predict_file,
    )
    manager = SMSDataManager(data_config)
    data = manager.load_all()
    manager.save_label_encoder()
    num_classes = len(manager.id2label)

    train_labels = data["train"]["label"].tolist()
    class_weights = compute_class_weights(train_labels, num_classes)

    if cfg.models.type == "mlp":
        model, train_loader, val_loader = _setup_mlp(cfg, num_classes, data)
    elif cfg.models.type == "bert":
        model, train_loader, val_loader = _setup_bert(cfg, num_classes, data)

    module = SMSClassificationModule(
        model=model,
        num_classes=num_classes,
        class_weights=class_weights,
        learning_rate=cfg.module.optimizer.learning_rate,
        loss_type=cfg.module.loss.type,
        focal_gamma=cfg.module.loss.focal_gamma,
        scheduler_eta_min=cfg.module.scheduler.eta_min,
    )

    # Loggers
    mlflow_logger = pl.pytorch.loggers.MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
        save_dir=cfg.logging.save_dir,
    )
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "stage", "train")
    mlflow_logger.experiment.log_param(mlflow_logger.run_id, "git_commit", _get_git_commit())

    loggers = [mlflow_logger]

    # Callbacks
    callbacks = [
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.pytorch.callbacks.RichModelSummary(max_depth=2),
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=cfg.callbacks.checkpoint.dirpath,
            filename=cfg.callbacks.checkpoint.filename,
            monitor=cfg.callbacks.checkpoint.monitor,
            mode=cfg.callbacks.checkpoint.mode,
            save_top_k=cfg.callbacks.checkpoint.save_top_k,
            every_n_epochs=cfg.callbacks.checkpoint.every_n_epochs,
        ),
        pl.pytorch.callbacks.EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            mode=cfg.callbacks.early_stopping.mode,
            patience=cfg.callbacks.early_stopping.patience,
            min_delta=cfg.callbacks.early_stopping.min_delta,
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.training.log_every_n_steps,
        accumulate_grad_batches=4,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    torch.save(module.model.state_dict(), cfg.models.output_path)


def _setup_mlp(cfg, num_classes: int, data: dict):
    """Setup MLP model with Word2Vec embeddings."""
    train_texts = data["train"]["text"].tolist()
    val_texts = data["val"]["text"].tolist()
    train_labels = data["train"]["label"].tolist()
    val_labels = data["val"]["label"].tolist()

    preprocessor = TextPreprocessor(language="russian")

    train_clean = [preprocessor.preprocess_text(t) for t in train_texts]
    val_clean = [preprocessor.preprocess_text(t) for t in val_texts]

    train_tokenized = [text.split() for text in train_clean]
    val_tokenized = [text.split() for text in val_clean]

    # Word2Vec
    w2v_cfg = cfg.models.word2vec
    embedder = Word2VecEmbedder(
        vector_size=w2v_cfg.vector_size,
        window=w2v_cfg.window,
        min_count=w2v_cfg.min_count,
        workers=w2v_cfg.workers,
        sg=w2v_cfg.sg,
    )
    train_embeddings = embedder.fit_transform(train_tokenized, epochs=w2v_cfg.epochs)
    val_embeddings = embedder.transform(val_tokenized)

    embedder.save(w2v_cfg.output_path)

    train_dataset = Word2VecDataset(train_embeddings, train_labels)
    val_dataset = Word2VecDataset(val_embeddings, val_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )

    model = MLPClassifier(
        input_dim=w2v_cfg.vector_size,
        num_classes=num_classes,
        dropout=cfg.models.dropout,
    )

    return model, train_loader, val_loader


def _setup_bert(cfg, num_classes: int, data: dict):
    """Setup BERT model."""
    train_texts = data["train"]["text"].tolist()
    val_texts = data["val"]["text"].tolist()
    train_labels = data["train"]["label"].tolist()
    val_labels = data["val"]["label"].tolist()

    tokenizer = BertTokenizerWrapper(
        pretrained_model=cfg.models.pretrained_model,
        max_length=cfg.models.max_length,
    )

    train_dataset = BertDataset(train_texts, train_labels, tokenizer=tokenizer)
    val_dataset = BertDataset(val_texts, val_labels, tokenizer=tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = BertClassifier(
        num_classes=num_classes,
        pretrained_model=cfg.models.pretrained_model,
        dropout=cfg.models.dropout,
        freeze_bert=cfg.models.freeze_bert,
    )

    return model, train_loader, val_loader
