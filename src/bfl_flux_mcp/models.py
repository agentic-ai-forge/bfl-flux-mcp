"""Pydantic models for API payloads and responses."""

from typing import Literal

from pydantic import BaseModel, Field

# --- Generation Models ---


class AspectRatio(BaseModel):
    """Valid aspect ratios for image generation."""

    RATIOS: tuple[str, ...] = ("1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21")


# --- Model Endpoint Mappings ---

GENERATION_MODELS = {
    "flux-dev": "v1/flux-dev",
    "flux-pro": "v1/flux-pro",
    "flux-pro-1.1": "v1/flux-pro-1.1",
    "flux-pro-1.1-ultra": "v1/flux-pro-1.1-ultra",
    "flux-2-pro": "v1/flux-2-pro",
    "flux-2-flex": "v1/flux-2-flex",
    "flux-2-max": "v1/flux-2-max",
}

EDIT_MODELS = {
    "kontext-pro": "v1/flux-kontext-pro",
    "kontext-max": "v1/flux-kontext-max",
    "fill-pro": "v1/flux-fill-pro",
}

VARIATION_MODELS = {
    "flux-dev": "v1/flux-dev",
    "flux-pro-1.1": "v1/flux-pro-1.1",
    "flux-pro-1.1-ultra": "v1/flux-pro-1.1-ultra",
}

EXPAND_ENDPOINT = "v1/flux-pro-1.0-expand"


# --- Type Aliases ---

GenerationModel = Literal[
    "flux-dev",
    "flux-pro",
    "flux-pro-1.1",
    "flux-pro-1.1-ultra",
    "flux-2-pro",
    "flux-2-flex",
    "flux-2-max",
]

EditModel = Literal["kontext-pro", "kontext-max", "fill-pro"]

VariationModel = Literal["flux-dev", "flux-pro-1.1", "flux-pro-1.1-ultra"]

AspectRatioLiteral = Literal["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"]

OutputFormat = Literal["png", "jpeg"]


# --- API Response Models ---


class TaskSubmission(BaseModel):
    """Response from task submission."""

    id: str
    polling_url: str | None = None


class TaskResult(BaseModel):
    """Response from task result polling."""

    status: str
    result: dict | list | None = None
    cost: float | None = None


class CreditsResponse(BaseModel):
    """Response from credits endpoint."""

    credits: float


class FinetuneInfo(BaseModel):
    """Information about a finetuned model."""

    name: str = "Unknown"
    status: str = "Unknown"
    id: str = Field(default="N/A", alias="id")


class FinetunesResponse(BaseModel):
    """Response from finetunes endpoint."""

    finetunes: list[FinetuneInfo] = []
