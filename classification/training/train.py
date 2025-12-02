import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from classification.data.dataset_loader import SMSDataManager
from classification.data.preprocess import TextPreprocessor
from classification.models.bert import BertClassifier, BertDataset, collate_fn
from classification.models.loss import compute_class_weights
from classification.models.mlp import MLPClassifier
from classification.models.word2vec import Word2VecDataset, Word2VecEmbedder
from classification.module.module import SMSClassificationModule
from classification.utils.data_util import ensure_data_downloaded
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    data_dir = Path(cfg.data.data_dir)
    required_paths = [
        str(data_dir / "train.csv"),
        str(data_dir / "val.csv"),
        str(data_dir / "test.csv"),
    ]

    if not ensure_data_downloaded(required_paths):
        print("Failed to download required data")
        sys.exit()

    L.seed_everything(cfg.seed)

    # Load data
    manager = SMSDataManager(Path(cfg.data.data_dir))
    data = manager.load_all()

    train_texts = data["train"]["text"].tolist()
    val_texts = data["val"]["text"].tolist()
    train_labels = data["train"]["label"].tolist()
    val_labels = data["val"]["label"].tolist()

    # Compute class weights
    class_weights = compute_class_weights(train_labels, cfg.model.num_classes)

    # Create model and dataloaders based on model type
    if cfg.model.type == "mlp":
        model, train_loader, val_loader = _setup_mlp(
            cfg, train_texts, val_texts, train_labels, val_labels
        )
    elif cfg.model.type == "bert":
        model, train_loader, val_loader = _setup_bert(
            cfg, train_texts, val_texts, train_labels, val_labels
        )

    # Create Lightning module
    module = SMSClassificationModule(
        model=model,
        num_classes=cfg.model.num_classes,
        class_weights=class_weights,
        learning_rate=cfg.training.learning_rate,
        loss_type=cfg.training.loss_type,
        focal_gamma=cfg.training.focal_gamma,
    )

    # Loggers
    loggers = [
        L.pytorch.loggers.MLFlowLogger(
            experiment_name=cfg.logging.experiment_name,
            tracking_uri=cfg.logging.tracking_uri,
            save_dir=cfg.logging.save_dir,
        ),
    ]

    # Callbacks
    callbacks = [
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        L.pytorch.callbacks.RichModelSummary(max_depth=2),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=cfg.callbacks.dirpath,
            filename=cfg.callbacks.filename,
            monitor="val_f1_weighted",
            mode="max",
            save_top_k=1,
            every_n_epochs=1,
        ),
        L.pytorch.callbacks.EarlyStopping(
            monitor="val_f1_weighted",
            mode="max",
            patience=cfg.callbacks.patience,
        ),
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        log_every_n_steps=cfg.training.log_every_n_steps,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save final model
    torch.save(module.model.state_dict(), cfg.model.output_path)


def _setup_mlp(cfg, train_texts, val_texts, train_labels, val_labels):
    """Setup MLP model with Word2Vec embeddings."""
    preprocessor = TextPreprocessor(language="russian")

    train_clean = [preprocessor.preprocess_text(t) for t in train_texts]
    val_clean = [preprocessor.preprocess_text(t) for t in val_texts]

    train_tokenized = [text.split() for text in train_clean]
    val_tokenized = [text.split() for text in val_clean]

    # Train Word2Vec
    embedder = Word2VecEmbedder(vector_size=cfg.model.vector_size)
    train_embeddings = embedder.fit_transform(train_tokenized)
    val_embeddings = embedder.transform(val_tokenized)

    # Save Word2Vec
    w2v_path = Path(cfg.model.output_path).parent / "word2vec.model"
    embedder.save(w2v_path)

    # Datasets
    train_dataset = Word2VecDataset(train_embeddings, train_labels)
    val_dataset = Word2VecDataset(val_embeddings, val_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )

    model = MLPClassifier(
        input_dim=cfg.model.vector_size,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )

    return model, train_loader, val_loader


def _setup_bert(cfg, train_texts, val_texts, train_labels, val_labels):
    """Setup BERT model."""
    train_dataset = BertDataset(train_texts, train_labels)
    val_dataset = BertDataset(val_texts, val_labels)

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
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        freeze_bert=cfg.model.freeze_bert,
    )

    return model, train_loader, val_loader


if __name__ == "__main__":
    main()
