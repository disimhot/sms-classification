import fire

from classification.inference.infer import infer
from classification.training.train import train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
        }
    )
