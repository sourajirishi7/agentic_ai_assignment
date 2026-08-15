"""Agentic workflow for deciding between schedule tools."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool

from .config import settings
from .date_utils import parse_date_text, parse_time_text
from .models import UpdateScheduleInput
from .tools import ScheduleTools


SYSTEM_PROMPT = """You are a schedule management assistant.
Never invent schedule events.
Use get_schedule when schedule information is required.
Use update_schedule when adding, modifying, or removing events.
For modifications to an existing event, retrieve it first if its ID is unknown.
Check for scheduling conflicts before adding or moving events.
Ask for clarification if required information is missing.
Never silently delete or overwrite events.
Keep responses concise and useful.
Base schedule answers on retrieved data.
"""


class ScheduleAgent:
    """Tool-using schedule agent with an offline deterministic fallback."""

    def __init__(self, schedule_tools: ScheduleTools | None = None) -> None:
        self.schedule_tools = schedule_tools or ScheduleTools()
        self.tools = self._build_langchain_tools()
        self.llm = self._build_llm()

    def _build_langchain_tools(self) -> list[Any]:
        @tool("get_schedule")
        def get_schedule_tool(query: str) -> str:
            """Retrieve schedule information for a natural-language query."""

            return self.schedule_tools.get_schedule(query)

        update_tool = StructuredTool.from_function(
            name="update_schedule",
            description="Add, update, or remove a schedule event.",
            func=lambda **kwargs: self.schedule_tools.update_schedule(**kwargs),
            args_schema=UpdateScheduleInput,
        )
        return [get_schedule_tool, update_tool]

    def _build_llm(self) -> object | None:
        if not settings.openai_api_key:
            return None
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=settings.llm_model, temperature=0).bind_tools(self.tools)
        except Exception:
            return None

    def invoke(self, user_input: str) -> str:
        """Return an assistant response for one user utterance."""

        if not user_input or not user_input.strip():
            return "Please enter a schedule question or request."
        if self.llm is None:
            return self._fallback_invoke(user_input)
        return self._llm_invoke(user_input)

    def _llm_invoke(self, user_input: str) -> str:
        messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_input)]
        tool_by_name = {tool_item.name: tool_item for tool_item in self.tools}
        try:
            ai_message = self.llm.invoke(messages)  # type: ignore[attr-defined]
            messages.append(ai_message)
            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                return str(getattr(ai_message, "content", ai_message))
            for call in tool_calls:
                name = call["name"]
                args = call.get("args", {})
                result = tool_by_name[name].invoke(args)
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            final = self.llm.invoke(messages)  # type: ignore[attr-defined]
            return str(getattr(final, "content", final))
        except Exception as exc:
            if self._is_quota_error(exc):
                return self._fallback_invoke(user_input)
            return f"Unable to complete the agent request: {exc}"

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        text = str(exc)
        return any(
            marker in text
            for marker in ("429", "insufficient_quota", "credit_balance_exhausted", "authentication", "api_key")
        )

    def _fallback_invoke(self, user_input: str) -> str:
        cleaned = user_input.lower().strip()
        if any(word in cleaned for word in ("add", "schedule", "create", "book")) and not cleaned.startswith(("what", "when", "am", "do", "show")):
            return self._fallback_add(user_input)
        if any(word in cleaned for word in ("cancel", "remove", "delete")):
            return self._fallback_remove(user_input)
        if any(word in cleaned for word in ("move", "reschedule", "change")):
            return self._fallback_update(user_input)
        return self.schedule_tools.get_schedule(user_input)

    def _fallback_add(self, text: str) -> str:
        title_match = re.search(r"(?:called|titled|named)\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = re.sub(r"^(add|schedule|create|book)\s+(a|an)?\s*", "", text, flags=re.IGNORECASE).strip()
            title = re.split(r"\s+(?:on|tomorrow|today|at)\b", title, maxsplit=1, flags=re.IGNORECASE)[0].strip() or "New Event"

        start_date, _ = parse_date_text(text)
        start_time = parse_time_text(text)
        if not start_date or not start_time:
            return "Please include a date and start time for the new event."
        end_time = (datetime.combine(date.today(), start_time) + timedelta(hours=1)).time()
        event_type = "meeting" if "meeting" in text.lower() else "task"
        return self.schedule_tools.update_schedule(
            action="add",
            title=title,
            date=start_date,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            description="Added from the command-line assistant.",
            location="Online",
        )

    def _fallback_remove(self, text: str) -> str:
        event_id = self._extract_event_id(text)
        if not event_id:
            matches = self.schedule_tools.get_schedule(text)
            return f"I need the event ID to remove it. Matching events:\n{matches}"
        return self.schedule_tools.update_schedule(action="remove", event_id=event_id)

    def _fallback_update(self, text: str) -> str:
        event_id = self._extract_event_id(text)
        start_time = parse_time_text(text)
        start_date, _ = parse_date_text(text)
        if not event_id:
            matches = self.schedule_tools.get_schedule(text)
            return f"I need the event ID to update it. Matching events:\n{matches}"
        kwargs: dict[str, Any] = {"action": "update", "event_id": event_id}
        if start_date:
            kwargs["date"] = start_date
        if start_time:
            kwargs["start_time"] = start_time
            kwargs["end_time"] = (datetime.combine(date.today(), start_time) + timedelta(hours=1)).time()
        if len(kwargs) == 2:
            return "Please include the new date, time, or details to update."
        return self.schedule_tools.update_schedule(**kwargs)

    def _extract_event_id(self, text: str) -> str | None:
        match = re.search(r"\bevent[_ -]?(\d{3,})\b", text, re.IGNORECASE)
        return f"event_{match.group(1)}" if match else None


def create_agent() -> ScheduleAgent:
    return ScheduleAgent()

