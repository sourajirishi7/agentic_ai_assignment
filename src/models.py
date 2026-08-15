"""Pydantic models for schedule entries and tool inputs."""

from __future__ import annotations

from datetime import date as Date
from datetime import time as Time
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

EventType = Literal["meeting", "workshop", "task", "appointment"]
Status = Literal["scheduled", "cancelled", "completed"]


class ScheduleEntry(BaseModel):
    """A single calendar item."""

    id: str
    title: str = Field(min_length=1, max_length=120)
    description: str = ""
    event_type: EventType
    date: Date
    start_time: Time
    end_time: Time
    location: str = "Online"
    status: Status = "scheduled"

    @field_validator("title", "description", "location", mode="before")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_time_order(self) -> "ScheduleEntry":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")
        return self


class UpdateScheduleInput(BaseModel):
    """Structured input accepted by the update_schedule tool."""

    action: Literal["add", "update", "remove"]
    event_id: str | None = None
    title: str | None = None
    description: str | None = None
    event_type: EventType | None = None
    date: Date | None = None
    start_time: Time | None = None
    end_time: Time | None = None
    location: str | None = None
    status: Status | None = None
    allow_conflict: bool = False
