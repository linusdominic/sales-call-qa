import pytest
from conftest import masked_segments
from fastapi.testclient import TestClient

from callqa import audio
from callqa.api import Dataset, create_app, tally


@pytest.fixture
def client(tmp_path, calls, cards, rubric) -> TestClient:
    (tmp_path / "call_001.mp3").write_bytes(b"ID3 not really audio")
    data = Dataset(
        rubric=rubric,
        calls={call.call_id: call for call in calls},
        cards={card.call_id: card for card in cards},
        transcripts={c.call_id: audio.to_json(masked_segments(c)) for c in calls},
        source="asr",
        model="claude-haiku-5-5",
        audio=tmp_path,
    )
    return TestClient(create_app(data))


def test_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "Sales call QA" in response.text


def test_totals_match_the_scorecards(client, cards):
    board = client.get("/api/dashboard").json()
    assert board["totals"] == tally(cards)
    assert board["totals"]["calls"] == len(cards)
    assert board["totals"]["meaningful"] + board["totals"]["not_meaningful"] == len(cards)
    assert sum(board["outcomes"].values()) == len(cards)
    assert board["run"] == {"source": "asr", "model": "claude-haiku-5-5"}


def test_officer_rows_add_up_to_the_totals(client):
    board = client.get("/api/dashboard").json()
    assert len(board["officers"]) == 8
    for key in ("calls", "reached", "meaningful", "flagged", "review"):
        assert sum(row[key] for row in board["officers"]) == board["totals"][key]


def test_campaign_filter_splits_the_calls(client, cards):
    cold = client.get("/api/dashboard?campaign=cold").json()["totals"]["calls"]
    warm = client.get("/api/dashboard?campaign=warm").json()["totals"]["calls"]
    assert cold + warm == len(cards) and cold and warm
    steps = {s["id"] for s in client.get("/api/dashboard?campaign=cold").json()["steps"]}
    assert "permission" in steps and "confirm_inquiry" not in steps


def test_officer_filter_narrows_totals_but_keeps_the_table(client):
    board = client.get("/api/dashboard?officer=Tom Becker").json()
    row = next(r for r in board["officers"] if r["officer"] == "Tom Becker")
    assert board["totals"]["calls"] == row["calls"]
    assert len(board["officers"]) == 8


def test_step_coverage_is_counted_over_reached_calls(client):
    board = client.get("/api/dashboard").json()
    assert all(
        0 <= s["covered"] <= s["total"] <= board["totals"]["reached"] for s in board["steps"]
    )


def test_call_lists(client, cards):
    everything = client.get("/api/calls").json()
    assert len(everything) == len(cards)
    assert everything[0]["started_at"] >= everything[-1]["started_at"]  # newest first
    review = client.get("/api/calls?show=review").json()
    assert review and all(call["review"] for call in review)
    yes = client.get("/api/calls?show=meaningful").json()
    no = client.get("/api/calls?show=not_meaningful").json()
    assert len(yes) + len(no) == len(cards) and all(call["meaningful"] for call in yes)
    assert client.get("/api/calls?show=bogus").status_code == 422


def test_call_detail_has_scorecard_transcript_and_truth(client, calls):
    call = next(c for c in calls if c.outcome == "conversation" and c.flags)
    detail = client.get(f"/api/calls/{call.call_id}").json()
    assert [s["id"] for s in detail["steps"]] == list(call.steps)
    covered = [s for s in detail["steps"] if s["verdict"] == "covered"]
    assert all(s["quote"] and s["verified"] for s in covered)
    assert [f["id"] for f in detail["flag_details"]] == call.flags
    assert len(detail["transcript"]) == len(call.turns)
    assert detail["truth"]["flags"] == call.flags
    assert client.get("/api/calls/nope").status_code == 404


def test_recording_is_served_only_when_on_disk(client):
    assert client.get("/api/calls/call_001").json()["audio"] == "/audio/call_001.mp3"
    assert client.get("/audio/call_001.mp3").status_code == 200
    assert client.get("/api/calls/call_002").json()["audio"] is None
    assert client.get("/audio/call_002.mp3").status_code == 404
    assert client.get("/audio/..%2Fsecret.mp3").status_code == 404
