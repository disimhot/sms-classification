from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import os

def train_model(model, tokenizer, datasets, compute_metrics, training_cfg):
    args = TrainingArguments(
        output_dir=training_cfg.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=training_cfg.learning_rate,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
        num_train_epochs=training_cfg.num_train_epochs,
        weight_decay=training_cfg.weight_decay,
        warmup_ratio=training_cfg.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model=training_cfg.metric_for_best_model,
        greater_is_better=training_cfg.greater_is_better,
        report_to="none",
        seed=training_cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    return trainer
