# import hydra
# from omegaconf import DictConfig
# from classification.model.model import load_model
# from classification.training.train import train_model


# @hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg):
    print("HI")

    # CSV_PATH = "data/my_dataset.csv"
    # MODEL_NAME = "deepvk/RuModernBERT-base"
    # MAX_LENGTH = 256

    # 1. читаем и делим
    # datasets_dict = load_and_split_dataset(CSV_PATH)

    # 2. токенизируем
    # tokenizer, tokenize_fn = get_tokenizer(MODEL_NAME, MAX_LENGTH)
    # datasets_tokenized = tokenize_datasets(datasets_dict, tokenize_fn)

    # model = load_model(cfg.model.pretrained_name, num_labels, id2label, label2id)

    # trainer = train_model(
    #     model=model,
    #     tokenizer=tokenizer,
    #     datasets=datasets,
    #     compute_metrics=compute_metrics,
    #     training_cfg=cfg.training,
    # )


if __name__ == "__main__":
    main()
