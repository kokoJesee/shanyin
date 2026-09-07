from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Scene = Literal[
    "theme_intent",
    "melody_feedback",
    "arrangement_variants",
    "practice_feedback",
    "creation_story",
    "teacher_report",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentRequest(StrictModel):
    schemaVersion: Literal["1"]
    requestId: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    sessionId: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    scene: Scene
    audience: Literal["child", "teacher"]
    context: dict[str, Any]

    @field_validator("context")
    @classmethod
    def limit_context(cls, value: dict[str, Any]) -> dict[str, Any]:
        encoded = str(value)
        if len(encoded) > 16_000:
            raise ValueError("context too large")
        blocked = {"name", "school", "phone", "address", "contact", "rawAudio", "pitchFrames"}
        if blocked.intersection(value):
            raise ValueError("context contains forbidden personal or raw recording fields")
        return value


class AgentResponse(StrictModel):
    schemaVersion: Literal["1"] = "1"
    scene: Scene
    text: str = Field(min_length=1, max_length=500)
    source: Literal["agent", "fallback", "offline"]
    story: str | None = Field(default=None, max_length=500)
    report: str | None = Field(default=None, max_length=500)


class NoteEvent(StrictModel):
    midi: int = Field(ge=21, le=108)
    note: str = Field(min_length=1, max_length=8)
    start: float = Field(ge=0, le=180)
    duration: float = Field(ge=0.04, le=12)
    velocity: float = Field(ge=0.05, le=1)
    confidence: float = Field(ge=0, le=1)


class ArrangementEvent(StrictModel):
    midi: int = Field(ge=21, le=108)
    note: str = Field(min_length=1, max_length=8)
    start: float = Field(ge=0, le=180)
    duration: float = Field(ge=0.04, le=12)
    velocity: float = Field(ge=0.05, le=1)
    role: Literal["melody", "bass", "chord"]


class KeyEstimate(StrictModel):
    tonic: str = Field(min_length=1, max_length=4)
    tonicPitchClass: int = Field(ge=0, le=11)
    mode: Literal["major", "minor"]
    confidence: float = Field(ge=0, le=1)


class Phrase(StrictModel):
    id: str = Field(min_length=1, max_length=32)
    label: str = Field(min_length=1, max_length=32)
    start: float = Field(ge=0, le=180)
    end: float = Field(ge=0, le=180)
    noteIndexes: list[int] = Field(max_length=128)


class Arrangement(StrictModel):
    schemaVersion: Literal["1"]
    tempoBpm: int = Field(ge=40, le=220)
    styleId: Literal["bright", "gentle", "strong", "spacious", "steady", "curious", "warm", "adventure", "dream", "natural"]
    variantId: Literal["faithful", "gentle", "playful"]
    key: KeyEstimate
    sourceMotif: list[NoteEvent] = Field(max_length=128)
    melody: list[NoteEvent] = Field(min_length=1, max_length=512)
    accompaniment: list[ArrangementEvent] = Field(min_length=1, max_length=2048)
    duration: float = Field(ge=1, le=180)
    phrases: list[Phrase] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_timeline(self) -> "Arrangement":
        if any(event.start + event.duration > self.duration + 1 for event in self.accompaniment):
            raise ValueError("event exceeds arrangement duration")
        return self


class RenderRequest(StrictModel):
    schemaVersion: Literal["1"]
    requestId: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    arrangement: Arrangement
    accompanimentGain: float = Field(default=0.75, ge=0, le=1)


class RenderResponse(StrictModel):
    schemaVersion: Literal["1"] = "1"
    fileId: str | None = None
    downloadUrl: str | None = None
    expiresAt: str

    @model_validator(mode="after")
    def require_location(self) -> "RenderResponse":
        if not self.fileId and not self.downloadUrl:
            raise ValueError("render response requires fileId or downloadUrl")
        return self
