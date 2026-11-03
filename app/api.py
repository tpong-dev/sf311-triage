"""FastAPI service for SF 311 triage.

    uvicorn app.api:app --port 8000        (with PYTHONPATH=src)

Endpoints:
    GET  /health      which backends are loaded and whether they are stubs
    POST /route       multipart: image file + lat, lon, timestamp (optional)
    POST /resolution  JSON: subtype, lat, lon, timestamp, source, category

Uploaded images are decoded in memory and never written to disk or logged.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.predictors import LATE_HOURS, PhotoRouter, ResolutionPredictor

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
Source = Literal["Mobile", "Web", "Phone", "Integrated Agency", "Twitter", "Email"]


# --------------------------------------------------------------------------- schemas


class LabelProb(BaseModel):
    label: str
    prob: float = Field(ge=0, le=1)


class RouteResponse(BaseModel):
    label: str
    top3: list[LabelProb]
    agency_hint: str | None = Field(None, description="Most common responsible agency for this category; a hint only")
    model: str
    is_stub: bool
    notes: list[str] = []


class ResolutionRequest(BaseModel):
    subtype: str = Field(..., examples=["garbage_and_debris"], description="service_subtype, e.g. garbage_and_debris")
    lat: float = Field(..., ge=-90, le=90, examples=[37.7749])
    lon: float = Field(..., ge=-180, le=180, examples=[-122.4194])
    timestamp: datetime | None = Field(None, description="Request time (local SF time); default now")
    source: Source = "Mobile"
    category: str | None = Field(None, description="service_name, used when the subtype is unknown")


class ResolutionResponse(BaseModel):
    median_hours: float
    p_late: float = Field(ge=0, le=1)
    late_threshold_hours: int = LATE_HOURS
    model: str
    is_stub: bool
    basis: str
    notes: list[str] = []


class HealthResponse(BaseModel):
    status: Literal["ok"]
    router: str
    router_is_stub: bool
    resolver: str
    resolver_is_stub: bool


# --------------------------------------------------------------------------- app


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests may inject predictors before startup; only load what is missing.
    if getattr(app.state, "router", None) is None:
        app.state.router = PhotoRouter.load()
    if getattr(app.state, "resolver", None) is None:
        app.state.resolver = ResolutionPredictor.load()
    yield


app = FastAPI(title="SF 311 Triage", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    r, s = request.app.state.router, request.app.state.resolver
    return HealthResponse(
        status="ok", router=r.name, router_is_stub=r.is_stub, resolver=s.name, resolver_is_stub=s.is_stub
    )


@app.post("/route", response_model=RouteResponse)
async def route(
    request: Request,
    image: UploadFile = File(..., description="JPEG, PNG or WebP photo, max 10 MB"),
    lat: float = Form(..., ge=-90, le=90),
    lon: float = Form(..., ge=-180, le=180),
    timestamp: datetime | None = Form(None),
) -> RouteResponse:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"unsupported image type {image.content_type!r}")
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        result = request.app.state.router.predict(data, lat, lon, timestamp)
    except (OSError, ValueError) as exc:  # PIL raises these for corrupt or non-image bytes
        raise HTTPException(422, "could not decode image") from exc
    finally:
        del data
    return RouteResponse(**result.to_dict())


@app.post("/resolution", response_model=ResolutionResponse)
def resolution(body: ResolutionRequest, request: Request) -> ResolutionResponse:
    result = request.app.state.resolver.predict(
        body.subtype, body.lat, body.lon, body.timestamp, body.source, body.category
    )
    return ResolutionResponse(**result.to_dict())
