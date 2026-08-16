from __future__ import annotations

from datetime import date, time, timedelta
from pathlib import Path
from uuid import uuid4

from src.data_manager import ScheduleDataManager, generate_sample_entries
from src.tools import ScheduleTools
from src.vector_store import ScheduleVectorStore


TEST_ROOT = Path("test_artifacts")


def make_tools():
    test_path = TEST_ROOT / uuid4().hex
    test_path.mkdir(parents=True, exist_ok=True)
    data_path = test_path / "schedule.json"
    manager = ScheduleDataManager(data_path)
    manager.save_entries(generate_sample_entries(date.today()))
    vector_store = ScheduleVectorStore(test_path / "chroma_db", collection_name="test_schedule")
    return ScheduleTools(manager, vector_store)


def test_adding_event():
    tools = make_tools()
    event_date = date.today() + timedelta(days=1)
    result = tools.update_schedule(
        action="add",
        title="Client Discussion",
        date=event_date,
        start_time=time(16, 0),
        end_time=time(17, 0),
        event_type="meeting",
        description="Discuss priorities.",
    )
    assert "Added" in result
    assert "Client Discussion" in tools.get_schedule(f"Show my schedule on {event_date.isoformat()}")


def test_updating_event():
    tools = make_tools()
    result = tools.update_schedule(action="update", event_id="event_001", start_time=time(18, 0), end_time=time(19, 0))
    assert "Updated" in result
    assert "6:00 PM - 7:00 PM" in tools.get_schedule("event_001")


def test_removing_event():
    tools = make_tools()
    result = tools.update_schedule(action="remove", event_id="event_001")
    assert "Removed" in result
    assert "Project Team Meeting" not in tools.get_schedule("Project Team Meeting")


def test_retrieving_events_by_date():
    tools = make_tools()
    target = date.today() + timedelta(days=1)
    result = tools.get_schedule(f"What do I have on {target.isoformat()}?")
    assert "Project Team Meeting" in result


def test_retrieving_events_by_event_type():
    tools = make_tools()
    result = tools.get_schedule("What appointments do I have?")
    assert "Appointment" in result
    assert "Dentist Appointment" in result


def test_detecting_conflicts():
    tools = make_tools()
    conflict_date = date.today() + timedelta(days=1)
    result = tools.update_schedule(
        action="add",
        title="Overlapping Meeting",
        date=conflict_date,
        start_time=time(14, 30),
        end_time=time(15, 30),
        event_type="meeting",
    )
    assert "scheduling conflict" in result
    assert "Project Team Meeting" in result


def test_checking_availability():
    tools = make_tools()
    result = tools.get_schedule("Am I free tomorrow afternoon?")
    assert "partially free" in result
    assert "Busy:" in result
    assert "Free:" in result


def test_updating_chromadb_after_modification():
    tools = make_tools()
    event_date = date.today() + timedelta(days=11)
    add_result = tools.update_schedule(
        action="add",
        title="Unique Vector Sync Review",
        date=event_date,
        start_time=time(9, 0),
        end_time=time(10, 0),
        event_type="meeting",
    )
    event_id = add_result.split("Event ID: ")[1].rstrip(".")
    assert any("Unique Vector Sync Review" in doc.page_content for doc in tools.vector_store.query("Unique Vector Sync Review"))
    tools.update_schedule(action="update", event_id=event_id, title="Renamed Vector Sync Review")
    docs = tools.vector_store.query("Renamed Vector Sync Review")
    assert any("Renamed Vector Sync Review" in doc.page_content for doc in docs)
    assert not any("Unique Vector Sync Review" in doc.page_content and doc.metadata.get("event_id") == event_id for doc in docs)
    tools.update_schedule(action="remove", event_id=event_id)
    assert not any(doc.metadata.get("event_id") == event_id for doc in tools.vector_store.query("Renamed Vector Sync Review"))


def test_example_natural_language_queries():
    tools = make_tools()
    queries = [
        "What do I have tomorrow?",
        "What meetings do I have next week?",
        "Am I free Friday afternoon?",
        "Show my schedule on August 15.",
    ]
    for query in queries:
        assert tools.get_schedule(query)
