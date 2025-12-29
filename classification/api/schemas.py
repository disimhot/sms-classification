from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for prediction endpoint."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to classify (1-100 items)",
        examples=[["Скидка 50% на все услуги до конца месяца"]],
    )


class PredictionItem(BaseModel):
    """Single prediction result."""

    text: str = Field(..., description="Original input text")
    label: str = Field(..., description="Predicted class label")
    label_id: int = Field(..., description="Predicted class ID")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: dict[str, float] = Field(..., description="Probabilities for all classes")


class PredictResponse(BaseModel):
    """Response body for prediction endpoint."""

    model_type: str = Field(..., description="Model used for prediction")
    predictions: list[PredictionItem]


class ModelStatus(BaseModel):
    """Status of a single model."""

    available: bool = Field(..., description="Whether model is loaded and ready")
    path: str = Field(..., description="Path to model weights")
    error: str | None = Field(None, description="Error message if not available")


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""

    status: str = Field(..., description="API status")
    models: dict[str, ModelStatus] = Field(..., description="Status of each model")


class ClassInfo(BaseModel):
    """Information about a classification class."""

    id: int
    name: str


class ModelsInfoResponse(BaseModel):
    """Response body for models info endpoint."""

    available_models: list[str] = Field(..., description="List of available model types")
    num_classes: int = Field(..., description="Number of classification classes")
    classes: list[ClassInfo] = Field(..., description="List of all classes")


class ErrorResponse(BaseModel):
    """Error response body."""

    detail: str = Field(..., description="Error message")
