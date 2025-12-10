from hydra import compose, initialize
from omegaconf import DictConfig


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load Hydra configuration.

    Args:
        overrides: List of config overrides, e.g. ["models=mlp", "training.max_epochs=5"]

    Returns:
        DictConfig: Loaded configuration
    """
    if overrides is None:
        overrides = []

    with initialize(config_path="../../conf"):
        return compose(config_name="conf", overrides=overrides)
