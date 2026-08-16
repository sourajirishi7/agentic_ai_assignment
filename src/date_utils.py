"""Small natural-language date and time helpers for schedule queries."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta


PERIODS: dict[str, tuple[time, time]] = {
    "morning": (time(8, 0), time(12, 0)),
    "afternoon": (time(12, 0), time(17, 0)),
    "evening": (time(17, 0), time(21, 0)),
}

WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})


@dataclass(frozen=True)
class QueryWindow:
    start_date: date | None = None
    end_date: date | None = None
    start_time: time | None = None
    end_time: time | None = None
    event_type: str | None = None
    availability: bool = False


def parse_time_text(text: str) -> time | None:
    """Parse simple time expressions such as 14:30, 2 PM, or 2:30pm."""

    text = text.strip().lower()
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return time(hour, minute)


def next_weekday(target_weekday: int, base: date | None = None, *, force_next: bool = False) -> date:
    base = base or date.today()
    days_ahead = (target_weekday - base.weekday()) % 7
    if days_ahead == 0 or force_next:
        days_ahead = days_ahead or 7
    return base + timedelta(days=days_ahead)


def parse_date_text(text: str, base: date | None = None) -> tuple[date | None, date | None]:
    """Parse common date/range phrases into start and end dates."""

    base = base or date.today()
    cleaned = text.lower()

    if "next week" in cleaned:
        start = base + timedelta(days=(7 - base.weekday()))
        return start, start + timedelta(days=6)
    if "this week" in cleaned:
        start = base - timedelta(days=base.weekday())
        return start, start + timedelta(days=6)
    if "next 30 days" in cleaned:
        return base, base + timedelta(days=30)
    if "tomorrow" in cleaned:
        target = base + timedelta(days=1)
        return target, target
    if "today" in cleaned:
        return base, base

    month_match = re.search(
        r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\s+(\d{1,2})(?:,\s*(\d{4}))?\b",
        cleaned,
    )
    if month_match:
        month = MONTHS[month_match.group(1)]
        day = int(month_match.group(2))
        year = int(month_match.group(3) or base.year)
        try:
            target = date(year, month, day)
            if not month_match.group(3) and target < base:
                target = date(year + 1, month, day)
            return target, target
        except ValueError:
            return None, None

    for weekday, index in WEEKDAYS.items():
        if re.search(rf"\bnext\s+{weekday}\b", cleaned):
            target = next_weekday(index, base, force_next=True)
            return target, target
        if re.search(rf"\b(?:this\s+)?{weekday}\b", cleaned):
            target = next_weekday(index, base)
            return target, target

    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", cleaned)
    if iso_match:
        try:
            target = datetime.strptime(iso_match.group(1), "%Y-%m-%d").date()
            return target, target
        except ValueError:
            return None, None

    return None, None


def parse_query_window(query: str, base: date | None = None) -> QueryWindow:
    """Infer filters and availability intent from natural language."""

    cleaned = query.lower().strip()
    start_date, end_date = parse_date_text(cleaned, base)
    start_time: time | None = None
    end_time: time | None = None

    for period, bounds in PERIODS.items():
        if period in cleaned:
            start_time, end_time = bounds
            break

    between = re.search(r"\bbetween\s+(.+?)\s+(?:and|to|-)\s+(.+?)(?:\?|$|\s+on|\s+tomorrow|\s+today)", cleaned)
    if between:
        start_time = parse_time_text(between.group(1))
        end_time = parse_time_text(between.group(2))
    elif " at " in cleaned:
        parsed = parse_time_text(cleaned.split(" at ", 1)[1])
        if parsed:
            start_time = parsed

    event_type = None
    for candidate in ("meeting", "workshop", "task", "appointment"):
        if candidate in cleaned or candidate + "s" in cleaned:
            event_type = candidate
            break

    availability = any(phrase in cleaned for phrase in ("am i free", "when am i free", "available", "availability"))
    return QueryWindow(start_date, end_date, start_time, end_time, event_type, availability)


def format_date(value: date) -> str:
    return f"{value.strftime('%A, %B')} {value.day}, {value.year}"


def format_time(value: time) -> str:
    return datetime.combine(date.today(), value).strftime("%I:%M %p").lstrip("0")


def overlaps(start_a: time, end_a: time, start_b: time, end_b: time) -> bool:
    return start_a < end_b and start_b < end_a
