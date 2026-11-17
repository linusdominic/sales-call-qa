"""Shared fixtures. Nothing here needs audio, the network or an API key."""

import pytest

from callqa import audio
from callqa.grader import Scorecard, finalize
from callqa.redact import redact
from callqa.rubric import Rubric
from callqa.synth import Call, generate


@pytest.fixture(scope="session")
def rubric() -> Rubric:
    return Rubric.load()


@pytest.fixture(scope="session")
def calls() -> list[Call]:
    return generate(count=80, seed=11)


def masked_segments(call: Call) -> list[audio.Segment]:
    return [
        audio.Segment(s.speaker, s.start, s.end, redact(s.text)[0])
        for s in audio.reference_segments(call)
    ]


def perfect_verdict(call: Call, rubric: Rubric) -> dict:
    """What a grader that made no mistakes would return for ``call``."""
    lines = masked_segments(call)
    agent_line = next(i for i, line in enumerate(lines) if line.speaker == "agent")
    quote = " ".join(lines[agent_line].text.split()[:4])
    conversation = call.outcome == "conversation"
    return {
        "outcome": call.outcome,
        "steps": [
            {
                "id": step.id,
                "verdict": ("covered" if call.steps[step.id] else "missing")
                if conversation
                else "not_applicable",
                "line": agent_line if conversation and call.steps[step.id] else -1,
                "quote": quote if conversation and call.steps[step.id] else "",
            }
            for step in rubric.steps_for(call.campaign)
        ],
        "flags": [{"id": flag, "line": agent_line, "quote": quote} for flag in call.flags],
        "confidence": "high",
        "summary": "A call.",
    }


def perfect_card(call: Call, rubric: Rubric) -> Scorecard:
    return finalize(
        perfect_verdict(call, rubric), call.call_id, call.campaign, masked_segments(call), rubric
    )


@pytest.fixture(scope="session")
def cards(calls, rubric) -> list[Scorecard]:
    return [perfect_card(call, rubric) for call in calls]
