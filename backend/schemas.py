"""
schemas.py — Pydantic request/response models for the Hawk-I HTTP API.

Named ``schemas`` rather than ``models`` on purpose: ``backend/`` is on
sys.path alongside the repo root, where ``models/`` holds the network weights.
A module called ``models.py`` there would collide with that directory as an
implicit namespace package, and which one won would depend on sys.path order.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentRequest(BaseModel):
    """POST /api/segment — one frame plus one box to measure."""

    frame_jpeg: str = Field(..., description="Base64-encoded JPEG, data-URI prefix optional")
    box: list[float] = Field(..., min_length=4, max_length=4,
                             description="Bounding box [x1, y1, x2, y2] in pixels")
    altitude_m: float = Field(..., ge=0, description="Drone altitude above ground, metres")


class QueryRequest(BaseModel):
    """POST /query — free-text defect query forwarded to the Jetson."""

    query: str = Field("", description="e.g. 'crack, rust stain and exposed rebar'")


class GPS(BaseModel):
    """GPS fix attached to a drone frame."""

    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0


class EdgeDetection(BaseModel):
    """One detection as emitted by the Jetson.

    ``class`` is a Python keyword, so the field is ``class_name`` with an alias.
    ``phrase`` is populated instead of ``class_name`` by the open-vocabulary
    detector; the processing worker accepts either.
    """

    class_name: str | None = Field(None, alias="class")
    phrase: str | None = None
    conf: float = 0.0
    box: list[float] = Field(default_factory=list)
    source: str | None = None

    model_config = {"populate_by_name": True}


class DronePayload(BaseModel):
    """The WebSocket message the Jetson sends on /ws/drone."""

    timestamp: float | None = None
    gps: GPS = Field(default_factory=GPS)
    yolo_detections: list[EdgeDetection] = Field(default_factory=list)
    gdino_detections: list[EdgeDetection] = Field(default_factory=list)
    frame_jpeg: str | None = None
