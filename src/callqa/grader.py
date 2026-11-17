"""Grade one call against the script, and check the grade before trusting it.

The model reads the transcript and returns a structured verdict: the call's outcome, each
script step as covered or missing, any compliance flags, and for every claim the line and
the exact words that support it. Code then verifies that those words are really in the
transcript, works out the business figures, and decides whether a person should look.
"""

import re
from dataclasses import asdict, dataclass, field

from callqa.audio import Segment
from callqa.llm import LLM
from callqa.rubric import Rubric

SYSTEM = """\
You grade recorded outreach calls for {company}, a mortgage lender, against its call script.

The transcript is numbered by line. AGENT is the loan officer. CUSTOMER is the person called.
It comes from speech recognition, so expect small errors. Personal details are masked, for \
example [PHONE]. A masked item still counts as having been said.

Decide three things.

1. OUTCOME. Exactly one of:
{outcomes}

2. SCRIPT STEPS. For each step below, "covered" or "missing", judged only on what the AGENT \
said. If the outcome is not "conversation", mark every step "not_applicable".
{steps}

3. COMPLIANCE FLAGS. List a flag only when the AGENT did what it describes. Nothing the \
CUSTOMER says raises a flag.
{flags}

Evidence: for every covered step and every flag, give the line number and a short exact \
quote copied from that AGENT line. For a missing or not applicable step use line -1 and an \
empty quote.

Confidence: "high" unless the transcript is garbled or a judgement was a close call. Finish \
with a one-sentence summary of the call for a sales manager."""

VERDICTS = ("covered", "missing", "not_applicable")


def system_prompt(rubric: Rubric, campaign: str) -> str:
    return SYSTEM.format(
        company=rubric.company,
        outcomes="\n".join(f"- {name}: {text}" for name, text in rubric.outcomes.items()),
        steps="\n".join(f"- {s.id}: {s.description}" for s in rubric.steps_for(campaign)),
        flags="\n".join(f"- {f.id}: {f.description}" for f in rubric.flags),
    )


def schema(rubric: Rubric, campaign: str) -> dict:
    evidence = {"line": {"type": "integer"}, "quote": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "outcome": {"type": "string", "enum": list(rubric.outcomes)},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "enum": [s.id for s in rubric.steps_for(campaign)],
                        },
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                        **evidence,
                    },
                    "required": ["id", "verdict", "line", "quote"],
                    "additionalProperties": False,
                },
            },
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": [f.id for f in rubric.flags]},
                        **evidence,
                    },
                    "required": ["id", "line", "quote"],
                    "additionalProperties": False,
                },
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "summary": {"type": "string"},
        },
        "required": ["outcome", "steps", "flags", "confidence", "summary"],
        "additionalProperties": False,
    }


def render(segments: list[Segment]) -> str:
    """The transcript as the model sees it: numbered lines with a time and a speaker."""
    lines = []
    for index, segment in enumerate(segments):
        minutes, seconds = divmod(int(segment.start), 60)
        lines.append(
            f"[{index}] {minutes:02d}:{seconds:02d} {segment.speaker.upper()}: {segment.text}"
        )
    return "\n".join(lines)


def _flat(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def locate(quote: str, line: int, segments: list[Segment]) -> int | None:
    """Find the AGENT line that contains ``quote``: the cited line first, then any other."""
    wanted = _flat(quote)
    if len(wanted) < 3:
        return None
    order = [line, *range(len(segments))] if 0 <= line < len(segments) else range(len(segments))
    for index in order:
        if segments[index].speaker == "agent" and wanted in _flat(segments[index].text):
            return index
    return None


@dataclass
class Evidence:
    id: str
    line: int | None  # the transcript line holding the quote, or None if it was not found
    quote: str
    verified: bool


@dataclass
class Scorecard:
    call_id: str
    campaign: str
    outcome: str
    steps: dict[str, str]  # step id -> covered | missing | not_applicable
    evidence: list[Evidence]
    flags: list[Evidence]
    confidence: str
    summary: str
    adherence: float | None  # share of script steps covered; None when not a conversation
    meaningful: bool
    review: list[str] = field(default_factory=list)  # why a person should look, if they should
    model: str = ""
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Scorecard":
        return cls(
            **{
                **data,
                "evidence": [Evidence(**item) for item in data["evidence"]],
                "flags": [Evidence(**item) for item in data["flags"]],
            }
        )

    @property
    def covered(self) -> set[str]:
        return {step for step, verdict in self.steps.items() if verdict == "covered"}


def finalize(
    raw: dict, call_id: str, campaign: str, segments: list[Segment], rubric: Rubric
) -> Scorecard:
    """Turn the model's verdict into a scorecard, trusting only what can be checked."""
    outcome = raw["outcome"]
    applicable = [step.id for step in rubric.steps_for(campaign)]
    conversation = outcome == rubric.meaningful_outcome
    given = {item["id"]: item for item in raw["steps"]}

    steps: dict[str, str] = {}
    evidence: list[Evidence] = []
    for step in applicable:
        item = given.get(step)
        if not conversation:
            steps[step] = "not_applicable"
        elif item is None or item["verdict"] != "covered":
            steps[step] = "missing"  # a step the model left out is not assumed to be covered
        else:
            steps[step] = "covered"
            line = locate(item["quote"], item["line"], segments)
            evidence.append(Evidence(step, line, item["quote"], line is not None))

    flags = []
    for item in raw["flags"]:
        line = locate(item["quote"], item["line"], segments)
        flags.append(Evidence(item["id"], line, item["quote"], line is not None))

    covered = {step for step, verdict in steps.items() if verdict == "covered"}
    review = []
    if flags:
        review.append("compliance flag")
    if raw["confidence"] != "high":
        review.append(f"{raw['confidence']} confidence")
    if any(not item.verified for item in [*evidence, *flags]):
        review.append("quoted evidence not found in transcript")

    return Scorecard(
        call_id=call_id,
        campaign=campaign,
        outcome=outcome,
        steps=steps,
        evidence=evidence,
        flags=flags,
        confidence=raw["confidence"],
        summary=raw["summary"].strip(),
        adherence=rubric.adherence(campaign, covered) if conversation else None,
        meaningful=rubric.is_meaningful(outcome, covered),
        review=review,
    )


async def grade(
    llm: LLM, model: str, rubric: Rubric, call_id: str, campaign: str, segments: list[Segment]
) -> Scorecard:
    result = await llm.complete(
        model=model,
        system=system_prompt(rubric, campaign),
        prompt=f"Campaign: {campaign} call\n\nTranscript:\n{render(segments)}",
        schema=schema(rubric, campaign),
        max_tokens=4000,
    )
    card = finalize(result.json(), call_id, campaign, segments, rubric)
    card.model, card.cost_usd, card.latency_s = model, result.cost_usd, round(result.latency_s, 2)
    return card
