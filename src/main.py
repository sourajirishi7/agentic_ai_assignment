"""Command-line entry point for the schedule assistant."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agent import ScheduleAgent


def main() -> None:
    print("====================================")
    print(" Agentic RAG Schedule Assistant")
    print("====================================")
    print()
    print("Type 'exit' to quit.")
    print()

    agent = ScheduleAgent()
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        print(f"\nAssistant: {agent.invoke(user_input)}\n")


if __name__ == "__main__":
    main()

