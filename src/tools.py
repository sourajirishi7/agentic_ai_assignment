"""The assistant's two schedule tools: get_schedule and update_schedule."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .data_manager import ScheduleDataManager
from .date_utils import PERIODS, format_date, format_time, overlaps, parse_query_window
from .models import ScheduleEntry, UpdateScheduleInput
from .vector_store import ScheduleVectorStore


class ScheduleTools:
    """Container for the only two tools exposed to the agent."""

    def __init__(self, data_manager: ScheduleDataManager | None = None, vector_store: ScheduleVectorStore | None = None) -> None:
        self.data_manager = data_manager or ScheduleDataManager()
        self.data_manager.ensure_sample_data()
        self.vector_store = vector_store or ScheduleVectorStore()
        self.vector_store.sync(self.data_manager.load_entries())

    def get_schedule(self, query: str) -> str:
        """Retrieve schedule information using date filters and vector search."""

        if not query or not query.strip():
            return "Please provide a schedule query."
        window = parse_query_window(query)
        entries = self._filter_entries(query)

        if window.availability:
            return self._availability_response(query, entries)

        if not entries:
            semantic_docs = self.vector_store.query(query)
            semantic_ids = {str(doc.metadata.get("event_id") or doc.metadata.get("id")) for doc in semantic_docs}
            entries = [entry for entry in self.data_manager.load_entries() if entry.id in semantic_ids]
            if window.event_type:
                entries = [entry for entry in entries if entry.event_type == window.event_type]

        if not entries:
            return "No scheduled events found for the requested period."
        return self._format_entries(entries, query)

    def update_schedule(self, **kwargs: Any) -> str:
        """Add, update, or remove one schedule event and keep ChromaDB synchronized."""

        try:
            payload = UpdateScheduleInput.model_validate(kwargs)
        except ValidationError as exc:
            return f"Invalid update request: {exc.errors()[0]['msg']}"

        if payload.action == "add":
            return self._add_event(payload)
        if payload.action == "update":
            return self._update_event(payload)
        if payload.action == "remove":
            return self._remove_event(payload.event_id)
        return "Invalid update action. Use add, update, or remove."

    def _filter_entries(self, query: str) -> list[ScheduleEntry]:
        window = parse_query_window(query)
        entries = [entry for entry in self.data_manager.load_entries() if entry.status == "scheduled"]
        if window.start_date:
            end_date = window.end_date or window.start_date
            entries = [entry for entry in entries if window.start_date <= entry.date <= end_date]
        if window.event_type:
            entries = [entry for entry in entries if entry.event_type == window.event_type]
        if window.start_time and window.end_time:
            entries = [entry for entry in entries if overlaps(entry.start_time, entry.end_time, window.start_time, window.end_time)]
        elif window.start_time:
            entries = [entry for entry in entries if entry.start_time <= window.start_time < entry.end_time]
        return sorted(entries, key=lambda item: (item.date, item.start_time))

    def _add_event(self, payload: UpdateScheduleInput) -> str:
        missing = [
            field
            for field in ("title", "date", "start_time", "end_time", "event_type")
            if getattr(payload, field) is None
        ]
        if missing:
            return f"Missing required fields for add: {', '.join(missing)}."
        conflicts = self.data_manager.find_conflicts(payload.date, payload.start_time, payload.end_time)
        if conflicts and not payload.allow_conflict:
            return self._conflict_message(conflicts)
        entry = ScheduleEntry(
            id=self.data_manager.next_event_id(),
            title=payload.title or "",
            description=payload.description or "",
            event_type=payload.event_type or "meeting",
            date=payload.date,
            start_time=payload.start_time,
            end_time=payload.end_time,
            location=payload.location or "Online",
            status=payload.status or "scheduled",
        )
        entries = self.data_manager.load_entries()
        entries.append(entry)
        self.data_manager.save_entries(entries)
        self.vector_store.upsert_entry(entry)
        return (
            f'Added "{entry.title}" on {format_date(entry.date)} '
            f"from {format_time(entry.start_time)} to {format_time(entry.end_time)}. Event ID: {entry.id}."
        )

    def _update_event(self, payload: UpdateScheduleInput) -> str:
        if not payload.event_id:
            return "Missing event_id for update."
        entries = self.data_manager.load_entries()
        index = next((i for i, entry in enumerate(entries) if entry.id == payload.event_id), None)
        if index is None:
            return f'No event found with ID "{payload.event_id}".'

        current = entries[index]
        update_data = current.model_dump()
        for field in ("title", "description", "event_type", "date", "start_time", "end_time", "location", "status"):
            value = getattr(payload, field)
            if value is not None:
                update_data[field] = value
        try:
            updated = ScheduleEntry.model_validate(update_data)
        except ValidationError as exc:
            return f"Invalid updated event: {exc.errors()[0]['msg']}"

        conflicts = self.data_manager.find_conflicts(
            updated.date,
            updated.start_time,
            updated.end_time,
            exclude_event_id=updated.id,
        )
        time_changed = (current.date, current.start_time, current.end_time) != (
            updated.date,
            updated.start_time,
            updated.end_time,
        )
        if conflicts and time_changed and not payload.allow_conflict:
            return self._conflict_message(conflicts)

        entries[index] = updated
        self.data_manager.save_entries(entries)
        self.vector_store.upsert_entry(updated)
        return (
            f'Updated "{updated.title}" on {format_date(updated.date)} '
            f"from {format_time(updated.start_time)} to {format_time(updated.end_time)}. Event ID: {updated.id}."
        )

    def _remove_event(self, event_id: str | None) -> str:
        if not event_id:
            return "Missing event_id for remove."
        entries = self.data_manager.load_entries()
        remaining = [entry for entry in entries if entry.id != event_id]
        if len(remaining) == len(entries):
            return f'No event found with ID "{event_id}".'
        removed = next(entry for entry in entries if entry.id == event_id)
        self.data_manager.save_entries(remaining)
        self.vector_store.delete_entry(event_id)
        return f'Removed "{removed.title}" from your schedule. Event ID: {event_id}.'

    def _availability_response(self, query: str, entries: list[ScheduleEntry]) -> str:
        window = parse_query_window(query)
        target_date = window.start_date or date.today()
        if window.start_time and window.end_time:
            start, end = window.start_time, window.end_time
        else:
            start, end = PERIODS["morning"][0], PERIODS["evening"][1]

        busy = sorted(
            [entry for entry in entries if entry.date == target_date and overlaps(entry.start_time, entry.end_time, start, end)],
            key=lambda item: item.start_time,
        )
        free_periods: list[tuple[time, time]] = []
        cursor = start
        for entry in busy:
            if cursor < entry.start_time:
                free_periods.append((cursor, entry.start_time))
            cursor = max(cursor, entry.end_time)
        if cursor < end:
            free_periods.append((cursor, end))

        label = f"{format_date(target_date)} from {format_time(start)} to {format_time(end)}"
        if not busy:
            return f"You are free {label}."
        if not free_periods:
            prefix = f"You are busy {label}."
        else:
            prefix = f"You are partially free {label}."
        busy_lines = "\n".join(f"- {format_time(e.start_time)} - {format_time(e.end_time)}: {e.title}" for e in busy)
        free_lines = "\n".join(f"- {format_time(s)} - {format_time(e)}" for s, e in free_periods) or "- No open periods."
        return f"{prefix}\n\nBusy:\n{busy_lines}\n\nFree:\n{free_lines}"

    def _format_entries(self, entries: list[ScheduleEntry], query: str) -> str:
        grouped: dict[date, list[ScheduleEntry]] = {}
        for entry in entries:
            grouped.setdefault(entry.date, []).append(entry)
        lines = [f"Found {len(entries)} scheduled event{'s' if len(entries) != 1 else ''}:"]
        for event_date, day_entries in sorted(grouped.items()):
            lines.append(f"\nSchedule for {format_date(event_date)}:")
            for entry in day_entries:
                lines.append(
                    f"{format_time(entry.start_time)} - {format_time(entry.end_time)}\n"
                    f"{entry.title}\n"
                    f"ID: {entry.id}\n"
                    f"Type: {entry.event_type.title()}\n"
                    f"Location: {entry.location}\n"
                    f"Status: {entry.status}"
                )
        return "\n".join(lines)

    def _conflict_message(self, conflicts: list[ScheduleEntry]) -> str:
        lines = ["There is a scheduling conflict:"]
        for conflict in conflicts:
            lines.append(
                f"- {conflict.title} from {format_time(conflict.start_time)} to {format_time(conflict.end_time)} "
                f"on {format_date(conflict.date)}."
            )
        lines.append("Would you like to schedule it anyway or choose another time?")
        return "\n".join(lines)


_default_tools: ScheduleTools | None = None


def get_default_tools() -> ScheduleTools:
    global _default_tools
    if _default_tools is None:
        _default_tools = ScheduleTools()
    return _default_tools


def get_schedule(query: str) -> str:
    """Public get_schedule tool."""

    return get_default_tools().get_schedule(query)


def update_schedule(**kwargs: Any) -> str:
    """Public update_schedule tool."""

    return get_default_tools().update_schedule(**kwargs)

