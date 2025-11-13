# sms-classification


## Стуктура проекта
project_root/
│
├── configs/
│   ├── model/
│   │   └── modernbert.yaml         # конфиг модели
│   ├── training/
│   │   └── default.yaml            # конфиг TrainingArguments
│   ├── data/
│   │   └── default.yaml            # пути, токенизация, пр.
│   └── main.yaml                   # главный hydra-конфиг (ссылается на все остальные)
│
├── classification/
│   ├── data/
│   │   ├── dataset_loader.py       # загрузка и токенизация датасета
│   │   └── preprocess.py           # текстовая предобработка
│   │
│   ├── models/
│   │   └── model.py     # обёртка над AutoModelForSequenceClassification
│   │
│   ├── training/
│   │   └── train.py                # логика Trainer
│   │
│   ├── inference/
│   │   └── infer.py                # Triton/онлайн-инференс
│   │
│   ├── metrics/
│   │   └── compute_metrics.py      # метрики (F1, accuracy и т.п.)
│   │
│   ├── utils/
│   │   └── logging_utils.py        # логирование, reproducibility и т.п.
│   │
│   └── main.py                     # точка входа, вызывает Hydra + Trainer
│
└── requirements.txt
