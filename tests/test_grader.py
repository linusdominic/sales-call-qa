import json

from conftest import masked_segments, perfect_verdict

from callqa.audio import Segment
from callqa.grader import finalize, grade, locate, render, schema, system_prompt
from callqa.llm import LLMResult, ModelRefused

LINES = [
    Segment("agent", 0.0, 4.0, "Hi Casey, this is James Carter from Harbor Home Lending."),
    Segment("customer", 4.5, 5.5, "Can you guarantee approval?"),
    Segment("agent", 6.0, 9.0, "I can't guarantee approval until underwriting reviews it."),
    Segment("agent", 65.0, 70.0, "You can reach me at [PHONE]."),
]


class FakeLLM:
    def __init__(self, reply: dict) -> None:
        self.reply, self.requests = reply, []

    async def complete(self, **request) -> LLMResult:
        self.requests.append(request)
        return LLMResult(json.dumps(self.reply), 900, 200, 0.0002, 1.234)


def verdict(**changes) -> dict:
    base = {"outcome": "conversation", "steps": [], "flags": [], "confidence": "high"}
    return {**base, "summary": " A call. ", **changes}


def covered(step: str, line: int, quote: str) -> dict:
    return {"id": step, "verdict": "covered", "line": line, "quote": quote}


def raised(flag: str, line: int, quote: str) -> dict:
    return {"id": flag, "verdict": "raised", "line": line, "quote": quote}


def test_prompt_lists_only_the_campaigns_steps(rubric):
    cold, warm = system_prompt(rubric, "cold"), system_prompt(rubric, "warm")
    assert "- permission:" in cold and "- confirm_inquiry:" not in cold
    assert "- confirm_inquiry:" in warm and "- permission:" not in warm
    assert all(f"- {flag.id}:" in cold for flag in rubric.flags)


def test_schema_restricts_ids_to_the_rubric(rubric):
    shape = schema(rubric, "cold")
    step_ids = shape["properties"]["steps"]["items"]["properties"]["id"]["enum"]
    assert step_ids == [step.id for step in rubric.steps_for("cold")]
    assert shape["properties"]["outcome"]["enum"] == list(rubric.outcomes)
    assert shape["additionalProperties"] is False


def test_render_numbers_lines_with_time_and_speaker():
    assert render(LINES).splitlines()[3] == "[3] 01:05 AGENT: You can reach me at [PHONE]."


def test_locate_ignores_case_and_punctuation():
    assert locate("this is james carter", 0, LINES) == 0
    assert locate("You can reach me at PHONE", 3, LINES) == 3


def test_locate_finds_the_quote_on_another_line_when_the_cited_one_is_wrong():
    assert locate("reach me at", 0, LINES) == 3
    assert locate("reach me at", 99, LINES) == 3


def test_locate_rejects_words_the_officer_never_said():
    assert locate("guaranteed to be approved", 2, LINES) is None
    assert locate("Can you guarantee approval?", 1, LINES) is None  # the contact said it
    assert locate("", 0, LINES) is None


def test_finalize_keeps_only_verdicts_it_can_use(rubric):
    raw = verdict(
        steps=[
            covered("greeting_identify", 0, "this is James Carter"),
            covered("closing", 3, "You can reach me at"),
            {"id": "purpose", "verdict": "missing", "line": -1, "quote": ""},
        ]
    )
    card = finalize(raw, "call_x", "cold", LINES, rubric)
    assert card.covered == {"greeting_identify", "closing"}
    assert card.steps["nmls_id"] == "missing"  # left out by the model: not assumed covered
    assert set(card.steps) == {step.id for step in rubric.steps_for("cold")}
    assert card.adherence == 0.2
    assert card.summary == "A call."
    assert card.review == []


def test_meaningful_is_decided_by_the_rule_not_the_model(rubric):
    steps = [
        covered("purpose", 0, "this is James Carter"),
        covered("qualify_goal", 0, "this is James Carter"),
    ]
    assert not finalize(verdict(steps=steps), "c", "cold", LINES, rubric).meaningful
    steps.append(covered("qualify_credit", 0, "this is James Carter"))
    assert finalize(verdict(steps=steps), "c", "cold", LINES, rubric).meaningful
    voicemail = verdict(outcome="voicemail", steps=steps)
    card = finalize(voicemail, "c", "cold", LINES, rubric)
    assert not card.meaningful and card.adherence is None
    assert set(card.steps.values()) == {"not_applicable"}


def test_unverifiable_quote_sends_the_call_to_review(rubric):
    raw = verdict(steps=[covered("nmls_id", 0, "my NMLS number is 517304")])
    card = finalize(raw, "c", "cold", LINES, rubric)
    assert card.evidence[0].verified is False and card.evidence[0].line is None
    assert card.review == ["quoted evidence not found in transcript"]


def test_flags_and_low_confidence_send_the_call_to_review(rubric):
    flag = raised("guaranteed_approval", 2, "I can't guarantee approval")
    card = finalize(verdict(flags=[flag], confidence="medium"), "c", "cold", LINES, rubric)
    assert card.review == ["compliance flag", "medium confidence"]
    assert card.flags[0].verified


def test_a_flag_marked_clear_is_not_a_flag(rubric):
    flags = [
        {"id": "guaranteed_approval", "verdict": "clear", "line": -1, "quote": ""},
        {"id": "rate_promise", "verdict": "clear", "line": 2, "quote": "I can't guarantee"},
        raised("high_pressure", 0, "this is James Carter"),
    ]
    card = finalize(verdict(flags=flags), "c", "cold", LINES, rubric)
    assert [flag.id for flag in card.flags] == ["high_pressure"]


def test_schema_asks_for_a_verdict_on_every_flag(rubric):
    flag = schema(rubric, "warm")["properties"]["flags"]["items"]
    assert flag["properties"]["verdict"]["enum"] == ["raised", "clear"]
    assert "verdict" in flag["required"]


async def test_grade_retries_with_more_room_when_output_runs_out(rubric):
    class RunsOut(FakeLLM):
        async def complete(self, **request) -> LLMResult:
            if len(self.requests) == 0:
                self.requests.append(request)
                raise ModelRefused("stopped with max_tokens")
            return await super().complete(**request)

    llm = RunsOut(verdict())
    card = await grade(llm, "claude-haiku-5-5", rubric, "call_x", "cold", LINES)
    assert [request["max_tokens"] for request in llm.requests] == [6000, 16000]
    assert card.outcome == "conversation"


def test_perfect_verdicts_reproduce_the_true_labels(calls, rubric):
    for call in calls:
        raw = perfect_verdict(call, rubric)
        card = finalize(raw, call.call_id, call.campaign, masked_segments(call), rubric)
        assert card.covered == {step for step, done in call.steps.items() if done}
        assert {flag.id for flag in card.flags} == set(call.flags)
        assert all(item.verified for item in [*card.evidence, *card.flags])


async def test_grade_sends_the_numbered_transcript_and_records_cost(rubric):
    llm = FakeLLM(verdict(steps=[covered("closing", 3, "reach me at")]))
    card = await grade(llm, "claude-haiku-5-5", rubric, "call_x", "cold", LINES)
    request = llm.requests[0]
    assert request["prompt"].startswith("Campaign: cold call")
    assert "[2] 00:06 AGENT: I can't guarantee approval" in request["prompt"]
    assert request["schema"] == schema(rubric, "cold")
    assert (card.model, card.cost_usd, card.latency_s) == ("claude-haiku-5-5", 0.0002, 1.23)
    assert card.covered == {"closing"}
