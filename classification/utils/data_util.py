import subprocess
from pathlib import Path

from dotenv import load_dotenv


def dvc_pull() -> bool:
    """
    Выполняет dvc pull с обработкой ошибок.

    Returns:
        bool: Успешно ли выполнена загрузка
    """
    load_dotenv()

    try:
        cmd = ["dvc", "pull"]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"DVC pull failed with error: {e.stderr}")
        return False


def ensure_data_downloaded(required_paths: list[str | Path]) -> bool:
    """
    Проверяет наличие путей и загружает недостающие через DVC.

    Args:
        required_paths: Список необходимых путей (папки/файлы)

    Returns:
        bool: Существуют ли все пути после выполнения
    """
    paths = [Path(p) for p in required_paths]
    missing_paths = [p for p in paths if not p.exists()]

    if not missing_paths:
        print("Все данные на месте.")
        return True

    print(f"Отсутствуют: {missing_paths}. Загружаю через DVC...")
    success = dvc_pull()

    if not success:
        return False

    # Проверяем ещё раз после pull
    still_missing = [p for p in paths if not p.exists()]
    if still_missing:
        print(f"После DVC pull всё ещё отсутствуют: {still_missing}")
        return False

    print("Данные успешно загружены.")
    return True
