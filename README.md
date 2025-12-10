# sms-classification

## Стуктура проекта

# Запустить всё (MLflow + train)

docker-compose up

# В фоне

docker-compose up -d

# С пересборкой образа

docker-compose up --build

# MLflow

docker-compose up mlflow

# Тренировка

docker-compose --profile train up

# Инференс

docker-compose --profile infer up
