"""Schedule JSON storage and sample data generation."""

from __future__ import annotations

import json
from datetime import date, time, timedelta
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from .config import settings
from .date_utils import overlaps
from .models import ScheduleEntry


class ScheduleDataManager:
    """Loads, validates, and writes schedule entries."""

    def __init__(self, schedule_path: Path | None = None) -> None:
        self.schedule_path = schedule_path or settings.schedule_path
        self.schedule_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_sample_data(self) -> None:
        if self.schedule_path.exists() and self.load_entries():
            return
        self.save_entries(generate_sample_entries())

    def load_entries(self) -> list[ScheduleEntry]:
        if not self.schedule_path.exists():
            return []
        try:
            raw = json.loads(self.schedule_path.read_text(encoding="utf-8"))
            return [ScheduleEntry.model_validate(item) for item in raw]
        except (json.JSONDecodeError, ValidationError, OSError) as exc:
            raise ValueError(f"Unable to load schedule data: {exc}") from exc

    def save_entries(self, entries: Iterable[ScheduleEntry]) -> None:
        payload = [entry.model_dump(mode="json") for entry in entries]
        self.schedule_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_event(self, event_id: str) -> ScheduleEntry | None:
        return next((entry for entry in self.load_entries() if entry.id == event_id), None)

    def next_event_id(self) -> str:
        max_id = 0
        for entry in self.load_entries():
            if entry.id.startswith("event_"):
                try:
                    max_id = max(max_id, int(entry.id.split("_", 1)[1]))
                except ValueError:
                    continue
        return f"event_{max_id + 1:03d}"

    def find_conflicts(
        self,
        event_date: date,
        start_time: time,
        end_time: time,
        *,
        exclude_event_id: str | None = None,
    ) -> list[ScheduleEntry]:
        conflicts: list[ScheduleEntry] = []
        for entry in self.load_entries():
            if entry.status != "scheduled" or entry.id == exclude_event_id or entry.date != event_date:
                continue
            if overlaps(entry.start_time, entry.end_time, start_time, end_time):
                conflicts.append(entry)
        return conflicts


def generate_sample_entries(base: date | None = None) -> list[ScheduleEntry]:
    """Generate realistic sample events across the next 30 days."""

    base = base or date.today()
    templates = [
        ("Project Team Meeting", "Discuss progress, blockers, and next steps.", "meeting", 1, "14:00", "15:00", "Online"),
        ("Product Roadmap Review", "Review upcoming milestones with stakeholders.", "meeting", 2, "10:00", "11:00", "Conference Room A"),
        ("Design Workshop", "Collaborative wireframe and prototype session.", "workshop", 3, "13:00", "15:00", "Studio"),
        ("Dentist Appointment", "Routine dental checkup.", "appointment", 4, "09:30", "10:15", "Smile Clinic"),
        ("Budget Planning Task", "Prepare quarterly budget notes.", "task", 5, "16:00", "17:00", "Office"),
        ("Client Discovery Call", "Capture client goals and constraints.", "meeting", 6, "11:00", "12:00", "Online"),
        ("AI Tools Workshop", "Hands-on session for workflow automation.", "workshop", 7, "15:00", "17:00", "Training Lab"),
        ("Write Status Report", "Draft and send weekly project update.", "task", 8, "09:00", "10:00", "Home Office"),
        ("Physical Therapy", "Follow-up mobility appointment.", "appointment", 9, "18:00", "19:00", "Health Center"),
        ("Vendor Sync", "Coordinate delivery and support timelines.", "meeting", 10, "12:30", "13:15", "Online"),
        ("Architecture Review", "Evaluate service boundaries and risks.", "meeting", 12, "14:30", "16:00", "Conference Room B"),
        ("Security Training", "Annual security awareness workshop.", "workshop", 13, "10:00", "12:00", "Auditorium"),
        ("Renew Insurance", "Compare renewal options and submit forms.", "task", 14, "17:30", "18:30", "Home Office"),
        ("Doctor Appointment", "Annual wellness visit.", "appointment", 15, "08:30", "09:30", "Medical Center"),
        ("Sprint Planning", "Plan tasks and commitments for the sprint.", "meeting", 16, "13:00", "14:00", "Online"),
        ("Data Cleanup", "Archive stale files and tag active assets.", "task", 18, "11:00", "12:30", "Office"),
        ("Leadership Check-in", "Review priorities and staffing needs.", "meeting", 20, "15:30", "16:15", "Online"),
        ("Negotiation Workshop", "Practice negotiation scenarios.", "workshop", 22, "09:00", "11:30", "Training Lab"),
        ("Car Service Appointment", "Scheduled oil change and inspection.", "appointment", 24, "14:00", "15:30", "AutoCare"),
        ("Prepare Demo", "Finalize notes and screenshots for demo.", "task", 26, "10:30", "12:00", "Home Office"),
        ("Retrospective", "Discuss what worked and what to improve.", "meeting", 28, "16:00", "17:00", "Online"),
        ("Portfolio Review", "Review personal goals and learning plan.", "task", 30, "18:00", "19:00", "Home Office"),
    ]
    entries: list[ScheduleEntry] = []
    for index, (title, description, event_type, offset, start, end, location) in enumerate(templates, start=1):
        entries.append(
            ScheduleEntry(
                id=f"event_{index:03d}",
                title=title,
                description=description,
                event_type=event_type,  # type: ignore[arg-type]
                date=base + timedelta(days=offset),
                start_time=start,
                end_time=end,
                location=location,
                status="scheduled",
            )
        )
    return entries

