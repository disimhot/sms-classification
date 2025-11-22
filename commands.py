import fire

# from classification.inference.infer import infer  # noqa: F401
from classification.training.train import train_model  # noqa: F401

if __name__ == "__main__":
    fire.Fire()
