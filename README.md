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

> ⚠️ **Важно**: В папке
> https://drive.google.com/drive/folders/1OpsoxTVF5wgCMFRi6KoErKTYVIRVvFOe?usp=sharing
> оставлены сэпмлы нечувствительных данных. За ключами доступа к S3/DVC
> обращайтесь лично к автору проекта тг @ElenaSergeevna.

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

### Loss функции

BERTподдерживает несколько функций потерь для работы с несбалансированными
данными:

- cross_entropy loss
- nll loss
- focal loss

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
├── .env.example              # Шаблон переменных окружения
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

### Настройка окружения

```bash
# Скопировать шаблон переменных окружения
cp .env.example .env

# Заполнить .env своими ключами (обратитесь к автору за доступом)
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
```

### Эксперименты с Loss функциями

```bash
# Cross Entropy (baseline)
uv run --project classification python commands.py train module.loss.type=cross_entropy

# Focal Loss с gamma=2.0 (default)
uv run --project classification python commands.py train module.loss.type=focal

# Focal Loss с разными значениями gamma
uv run --project classification python commands.py train module.loss.type=focal module.loss.focal_gamma=1.0
uv run --project classification python commands.py train module.loss.type=focal module.loss.focal_gamma=2.0
uv run --project classification python commands.py train module.loss.type=focal module.loss.focal_gamma=3.0
uv run --project classification python commands.py train module.loss.type=focal module.loss.focal_gamma=5.0

# Комбинация модели и loss
uv run --project classification python commands.py train models=mlp module.loss.type=focal module.loss.focal_gamma=2.0
```

### Другие параметры обучения

```bash
# Изменение learning rate
uv run --project classification python commands.py train module.optimizer.learning_rate=1e-4

# Изменение batch size и epochs
uv run --project classification python commands.py train training.batch_size=16 training.max_epochs=100

# Комбинация нескольких параметров
uv run --project classification python commands.py train \
    models=bert \
    module.loss.type=focal \
    module.loss.focal_gamma=2.0 \
    training.batch_size=16 \
    training.max_epochs=50
```

### Конфигурация

Параметры обучения задаются через Hydra конфиги в папке `conf/`:

**training/default.yaml**

```yaml
max_epochs: 50
batch_size: 32
log_every_n_steps: 10
```

**module/default.yaml**

```yaml
loss:
  type: cross_entropy # cross_entropy | focal | nll
  focal_gamma: 2.0 # только для focal loss

optimizer:
  learning_rate: 1e-3

scheduler:
  eta_min: 1e-7
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

### Тестирование на test dataset

Оценка качества модели на тестовой выборке:

```bash
# BERT модель (по умолчанию)
uv run --project classification python commands.py infer

# MLP модель
uv run --project classification python commands.py infer models=mlp
```

Результаты дополнительно выводятся в консоль:

```
=== Test Results ===
test_loss: 0.4523
test_accuracy: 0.8934
test_f1_weighted: 0.8912
test_f1_macro: 0.8567
test_auroc_weighted: 0.9823
test_auroc_macro: 0.9756
```

### Batch предсказания на новых данных

Предсказания для файла `data/predict.csv`:

```bash
# С BERT моделью
uv run --project classification python commands.py predict

# С MLP моделью
uv run --project classification python commands.py predict models=mlp
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
- `POST /predict` — предсказание для одного сообщения

#### Примеры запросов

```bash
# Health check
curl http://localhost:8000/health

# Список моделей
curl http://localhost:8000/models

# Предсказание с BERT
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Скидка 50% на ремонт компьютеров!", "model_type": "bert"}'

# Предсказание с MLP
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Одобрен кредит на 500000 рублей", "model_type": "mlp"}'
```

#### Пример ответа

```json
{
  "text": "Скидка 50% на ремонт компьютеров!",
  "predicted_class": "Товары/Электроника",
  "confidence": 0.9234,
  "probabilities": {
    "Товары/Электроника": 0.9234,
    "Займы": 0.0321,
    "Страховые услуги": 0.0156,
    ...
  }
}
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

### CI/CD

Проект использует GitHub Actions для автоматической проверки качества кода.

#### Workflow

При каждом push и pull request в main запускается проверка pre-commit хуков:

- Ruff (форматирование и линтинг)
- Другие настроенные хуки

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

## Лицензия

MIT

MLflow by [MLflow](https://mlflow.org/) ![MLflow.png](data_example/MLflow.png)

Server by [FastAPI](https://fastapi.tiangolo.com/)
![SMS-Classification-API-Swagger-UI.png](data_example/SMS-Classification-API-Swagger-UI.png)
![SMS-Classification-API-Swagger-UI1.png](data_example/SMS-Classification-API-Swagger-UI1.png)
