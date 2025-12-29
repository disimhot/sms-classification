import fire

from classification.inference.infer import infer
from classification.training.train import train


def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    import uvicorn

    uvicorn.run(
        "classification.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "serve": serve,
        }
    )
