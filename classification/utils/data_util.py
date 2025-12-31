import subprocess
from pathlib import Path

from dotenv import load_dotenv


def dvc_pull() -> bool:
    """
    Make DVC pull.

    Returns:
        bool: Success of DVC pull
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
    Ensure that all required data is downloaded.

    Args:
        required_paths: List of paths to check

    Returns:
        bool: True if all required data is downloaded
    """
    paths = [Path(p) for p in required_paths]
    missing_paths = [p for p in paths if not p.exists()]

    if not missing_paths:
        print("All required data is downloaded.")
        return True

    print(f"Absent data: {missing_paths}")
    success = dvc_pull()

    if not success:
        return False

    still_missing = [p for p in paths if not p.exists()]
    if still_missing:
        print(f"Still missing data: {still_missing}")
        return False

    print("All required data is downloaded.")
    return True
