# SMS Classification

Проект классификации SMS-сообщений по категориям с использованием моделей
машинного обучения.

## Описание проекта

### Задача

Задача заключается в автоматической категоризации сообщений на заранее
определённые категории (например, «Товары/Электроника», «Займы», «Страховые
услуги» и др.) для последующего анализа маркетинговых кампаний и модерации
сообщений.

### Данные

- **Источник**: Размеченный датасет SMS-сообщений
- **Формат**: CSV с полями `text` (текст сообщения) и `result` (категория)
- **Разбиение**: train/val/test
- **Хранение**: DVC (Data Version Control)

### Модели

Проект поддерживает две архитектуры:

1. **MLP (Baseline)**
   - Word2Vec эмбеддинги (Gensim)
   - Многослойный перцептрон
   - Быстрое обучение, подходит для экспериментов

2. **BERT**
   - Предобученная модель DeepPavlov/rubert-base-cased
   - Fine-tuning на задачу классификации
   - Лучшее качество, но требует больше ресурсов

### Предобработка текста

- Удаление HTML-тегов
- Обработка emoji
- Удаление пунктуации и кавычек
- Токенизация (NLTK)
- Лемматизация (Stanza)
- Удаление стоп-слов

### Метрики

- F1-score (weighted, macro)
- ROC-AUC (weighted, macro)
- Accuracy

## Структура проекта

```
sms-classification/
├── classification/
│   ├── api/
│   │   ├── app.py
│   │   └── schemas.py
│   ├── data/
│   │   ├── dataset_loader.py
│   │   ├── label_encoder.py
│   │   └── preprocess.py
│   ├── inference/
│   │   ├── infer.py
│   │   └── predict.py
│   ├── models/
│   │   ├── bert.py
│   │   ├── mlp.py
│   │   ├── word2vec.py
│   │   └── loss.py
│   ├── module/
│   │   └── module.py
│   ├── training/
│   │   └── train.py
│   ├── utils/
│   └── pyproject.toml
├── conf/
│   ├── conf.yaml
│   ├── callbacks/
│   ├── data/
│   ├── inference/
│   ├── logging/
│   ├── models/
│   ├── module/
│   └── training/
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── predict.csv
├── output/
│   ├── models/
│   ├── predictions/
│   ├── checkpoints/
│   └── logs/
├── commands.py               # CLI точка входа
├── docker-compose.yaml
├── Dockerfile
└── data.dvc
```

## Setup

### Требования

- Python 3.13+
- uv (менеджер пакетов)

### Установка uv

```bash

pip install uv
```

### Клонирование и настройка

```bash
# Клонировать репозиторий
git clone https://github.com/disimhot/sms-classification.git
cd sms-classification

# Установить зависимости
cd classification
uv sync
cd ..

# Настроить pre-commit хуки
uv run --project classification pre-commit install
```

### Загрузка данных

```bash
# Настроить DVC remote (если нужно)
dvc remote modify --local myremote access_key_id <YOUR_KEY>
dvc remote modify --local myremote secret_access_key <YOUR_SECRET>

# Загрузить данные
dvc pull
```

## Train

### Запуск MLflow

Перед обучением запустите MLflow сервер для логирования:

```bash
# Вариант 1: напрямую
mlflow server --host 127.0.0.1 --port 8080

# Вариант 2: через Docker
docker-compose up mlflow
```

MLflow UI будет доступен по адресу http://127.0.0.1:8080

### Обучение модели

```bash
# BERT модель (по умолчанию)
uv run --project classification python commands.py train

# MLP модель
uv run --project classification python commands.py train models=mlp

# С переопределением параметров
uv run --project classification python commands.py train "['training.batch_size=8', 'training.max_epochs=10']"
```

### Конфигурация

Параметры обучения задаются через Hydra конфиги в папке `conf/`:

**training/default.yaml**

```yaml
max_epochs: 50
batch_size: 32
log_every_n_steps: 10
```

**models/bert.yaml**

```yaml
type: bert
pretrained_model: DeepPavlov/rubert-base-cased
max_length: 128
dropout: 0.3
freeze_bert: true
output_path: output/models/bert.pt
```

**models/mlp.yaml**

```yaml
type: mlp
dropout: 0.3
output_path: output/models/mlp.pt
```

### Результаты обучения

- Веса модели: `output/models/bert.pt` или `output/models/mlp.pt`
- Чекпоинты: `data/checkpoints/`
- Label encoder: `data/label_encoder.json`
- Метрики: MLflow UI

## Inference

### Batch inference

```bash
# Предсказания на файле
uv run --project classification python commands.py infer

# С указанием модели
uv run --project classification python commands.py infer models=mlp
```

Результаты сохраняются в `output/predictions/predictions.csv`

### API сервис

```bash
# Запуск сервера
uv run --project classification python commands.py serve

# С параметрами
uv run --project classification python commands.py serve --port 8000 --reload
```

#### Endpoints

- `GET /health` — проверка состояния
- `GET /models` — список загруженных моделей
- `POST /predict` — предсказание

#### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Скидка 50% на ремонт компьютеров!", "model_type": "bert"}'
```

## Docker

### Сборка и запуск

```bash
# Запуск API
docker-compose up api

# Запуск обучения
docker-compose --profile train up train

# Запуск всего (API + MLflow)
docker-compose up api mlflow
```

### Порты

- API: 8000
- MLflow: 8080

## Разработка

### Code quality

Проект использует pre-commit с Ruff для форматирования и линтинга:

```bash
# Запуск проверок вручную
uv run --project classification pre-commit run --all-files

# Форматирование
uv run --project classification ruff format .

# Линтинг
uv run --project classification ruff check . --fix
```

### Добавление зависимостей

```bash
cd classification
uv add <package>
```

## Технологии

| Компонент                 | Технология        |
| ------------------------- | ----------------- |
| Пакетный менеджер         | uv                |
| Фреймворк обучения        | PyTorch Lightning |
| Конфигурация              | Hydra             |
| Версионирование данных    | DVC               |
| Логирование экспериментов | MLflow            |
| API                       | FastAPI           |
| Code quality              | Ruff, pre-commit  |
| Контейнеризация           | Docker            |
