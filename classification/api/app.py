from contextlib import asynccontextmanager
from typing import Literal

from classification.api.schemas import (
    ClassInfo,
    ErrorResponse,
    HealthResponse,
    ModelsInfoResponse,
    ModelStatus,
    PredictionItem,
    PredictRequest,
    PredictResponse,
)
from classification.inference.predict import (
    LabelEncoderNotFoundError,
    ModelNotFoundError,
    Predictor,
)
from fastapi import FastAPI, HTTPException, status


class PredictorManager:
    """Manages predictor instances for different model types."""

    def __init__(self):
        self.predictors: dict[str, Predictor | None] = {
            "bert": None,
            "mlp": None,
        }
        self.errors: dict[str, str | None] = {
            "bert": None,
            "mlp": None,
        }
        self.id2label: dict[int, str] = {}
        self.label2id: dict[str, int] = {}

    def load_all(self) -> None:
        """Attempt to load all model types."""
        for model_type in ["bert", "mlp"]:
            self._load_model(model_type)

    def _load_model(self, model_type: str) -> None:
        """Load a single model type."""
        try:
            predictor = Predictor(model_type=model_type)
            predictor.load()
            self.predictors[model_type] = predictor
            self.errors[model_type] = None

            # Store label mappings from first successful load
            if not self.id2label:
                self.id2label = predictor.id2label
                self.label2id = predictor.label2id

            print(f"✓ {model_type.upper()} model loaded successfully")

        except (ModelNotFoundError, LabelEncoderNotFoundError) as e:
            self.predictors[model_type] = None
            self.errors[model_type] = str(e)
            print(f"✗ {model_type.upper()} model not available: {e}")

        except Exception as e:
            self.predictors[model_type] = None
            self.errors[model_type] = f"Unexpected error: {e}"
            print(f"✗ {model_type.upper()} model failed to load: {e}")

    def get_predictor(self, model_type: str) -> Predictor:
        """Get predictor for model type or raise HTTPException."""
        predictor = self.predictors.get(model_type)

        if predictor is None:
            error_msg = self.errors.get(model_type, "Model not available")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error_msg,
            )

        return predictor

    def get_available_models(self) -> list[str]:
        """Return list of available model types."""
        return [k for k, v in self.predictors.items() if v is not None]


manager = PredictorManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("Starting SMS Classification API")
    manager.load_all()

    available = manager.get_available_models()
    if available:
        print(f"Ready! Available models: {', '.join(available)}")
    else:
        print("Warning: No models available. Train models first.")

    yield


app = FastAPI(
    title="SMS Classification API",
    description="API for classifying SMS messages using BERT or MLP models",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check API health and model availability",
)
async def health() -> HealthResponse:
    """Check API health and model availability."""
    models_status = {}

    for model_type in ["bert", "mlp"]:
        predictor = manager.predictors.get(model_type)

        if predictor is not None:
            models_status[model_type] = ModelStatus(
                available=True,
                path=predictor.get_model_path(),
                error=None,
            )
        else:
            # Get path from config for unavailable models
            try:
                temp_predictor = Predictor(model_type=model_type)
                path = temp_predictor.get_model_path()
            except Exception:
                path = "unknown"

            models_status[model_type] = ModelStatus(
                available=False,
                path=path,
                error=manager.errors.get(model_type),
            )

    return HealthResponse(status="healthy", models=models_status)


@app.get(
    "/models",
    response_model=ModelsInfoResponse,
    tags=["Models"],
    summary="Get models info",
    description="Get information about available models and classes",
)
async def models_info() -> ModelsInfoResponse:
    """Get information about available models and classes."""
    if not manager.id2label:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded. Please train at least one model first.",
        )

    classes = [ClassInfo(id=id_, name=name) for id_, name in sorted(manager.id2label.items())]

    return ModelsInfoResponse(
        available_models=manager.get_available_models(),
        num_classes=len(manager.id2label),
        classes=classes,
    )


@app.post(
    "/predict/{model_type}",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict with specific model",
    description="Classify texts using BERT or MLP model",
    responses={
        200: {"description": "Successful prediction"},
        503: {"model": ErrorResponse, "description": "Model not available"},
    },
)
async def predict(
    model_type: Literal["bert", "mlp"],
    request: PredictRequest,
) -> PredictResponse:
    """Classify texts using the specified model."""
    predictor = manager.get_predictor(model_type)

    try:
        results = predictor.predict(request.texts)

        predictions = [
            PredictionItem(
                text=r["text"],
                label=r["label"],
                label_id=r["label_id"],
                confidence=r["confidence"],
                probabilities=r["probabilities"],
            )
            for r in results
        ]

        return PredictResponse(
            model_type=model_type,
            predictions=predictions,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}",
        ) from e
