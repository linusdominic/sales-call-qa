"""The dashboard: what a sales manager sees, built only from the graded scorecards.

Three views share one filter (campaign and loan officer): the headline counts, a table per
loan officer, and the list of calls. Opening a call shows the scorecard next to the
transcript, with the quote behind every verdict, and the recording if it is on disk.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from callqa import store
from callqa.grader import Scorecard
from callqa.rubric import Rubric
from callqa.synth import Call

STATIC = Path(__file__).parent / "static"


@dataclass
class Dataset:
    rubric: Rubric
    calls: dict[str, Call]
    cards: dict[str, Scorecard]
    transcripts: dict[str, list[dict]]  # call id -> masked transcript lines
    source: str
    model: str
    audio: Path

    @classmethod
    def load(cls, root: Path = store.ROOT) -> "Dataset":
        results = root / "results"
        wanted = os.environ.get("CALLQA_RUN")  # for example "clean.haiku-5-5"
        paths = sorted(results.glob("scorecards.*.jsonl"), key=lambda p: ".asr." not in p.name)
        if wanted:
            paths = [results / f"scorecards.{wanted}.jsonl"]
        if not paths or not paths[0].exists():
            raise FileNotFoundError("No scorecards found. Run `callqa grade` first.")
        _, source, model = paths[0].stem.split(".", 2)
        calls = [Call.from_json(row) for row in store.read(root / "data" / "calls.jsonl")]
        cards = [Scorecard.from_json(row) for row in store.read(paths[0])]
        if source == "asr":
            rows = store.read(root / "data" / "transcripts.jsonl")
            transcripts = {row["call_id"]: row["masked"] for row in rows}
        else:
            from callqa import audio
            from callqa.redact import redact

            transcripts = {
                call.call_id: [
                    {**line, "text": redact(line["text"])[0]}
                    for line in audio.to_json(audio.reference_segments(call))
                ]
                for call in calls
            }
        return cls(
            rubric=Rubric.load(),
            calls={call.call_id: call for call in calls},
            cards={card.call_id: card for card in cards},
            transcripts=transcripts,
            source=source,
            model=f"claude-{model}",
            audio=root / "data" / "audio",
        )

    def select(self, campaign: str = "all", officer: str = "") -> list[Scorecard]:
        """The graded calls that match the filter, oldest first."""
        chosen = [
            card
            for card in self.cards.values()
            if (campaign == "all" or card.campaign == campaign)
            and (not officer or self.calls[card.call_id].officer == officer)
        ]
        return sorted(chosen, key=lambda card: self.calls[card.call_id].started_at)


def tally(cards: list[Scorecard]) -> dict:
    """The counts shown for any group of calls."""
    reached = [card for card in cards if card.adherence is not None]
    meaningful = sum(card.meaningful for card in cards)
    return {
        "calls": len(cards),
        "reached": len(reached),
        "meaningful": meaningful,
        "not_meaningful": len(cards) - meaningful,
        "meaningful_rate": meaningful / len(cards) if cards else None,
        "adherence": sum(c.adherence for c in reached) / len(reached) if reached else None,
        "flagged": sum(bool(card.flags) for card in cards),
        "review": sum(bool(card.review) for card in cards),
    }


def row(card: Scorecard, call: Call) -> dict:
    return {
        "call_id": card.call_id,
        "started_at": call.started_at,
        "officer": call.officer,
        "customer": call.customer,
        "campaign": card.campaign,
        "outcome": card.outcome,
        "meaningful": card.meaningful,
        "adherence": card.adherence,
        "flags": [flag.id for flag in card.flags],
        "review": card.review,
        "summary": card.summary,
    }


def create_app(data: Dataset | None = None) -> FastAPI:
    app = FastAPI(title="Sales call QA", docs_url=None, redoc_url=None)
    state: dict[str, Dataset] = {}

    def dataset() -> Dataset:
        if "data" not in state:
            try:
                state["data"] = data or Dataset.load()
            except FileNotFoundError as error:
                raise HTTPException(503, str(error)) from error
        return state["data"]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/dashboard")
    def dashboard(campaign: str = "all", officer: str = "") -> dict:
        d = dataset()
        cards = d.select(campaign, officer)
        everyone = d.select(campaign)
        officers = sorted({call.officer for call in d.calls.values()})
        reached = [card for card in cards if card.adherence is not None]
        steps = [
            {
                "id": step.id,
                "label": step.label,
                "covered": sum(card.steps.get(step.id) == "covered" for card in reached),
                "total": sum(step.id in card.steps for card in reached),
            }
            for step in d.rubric.steps
        ]
        outcomes: dict[str, int] = {}
        for card in cards:
            outcomes[card.outcome] = outcomes.get(card.outcome, 0) + 1
        return {
            "company": d.rubric.company,
            "run": {"source": d.source, "model": d.model},
            "officer_names": officers,
            "totals": tally(cards),
            "officers": [
                {
                    "officer": name,
                    **tally([c for c in everyone if d.calls[c.call_id].officer == name]),
                }
                for name in officers
            ],
            "steps": [step for step in steps if step["total"]],
            "outcomes": outcomes,
            "flags": {
                flag.id: {
                    "label": flag.label,
                    "calls": sum(flag.id in {f.id for f in card.flags} for card in cards),
                }
                for flag in d.rubric.flags
            },
        }

    @app.get("/api/calls")
    def calls(campaign: str = "all", officer: str = "", show: str = "all") -> list[dict]:
        d = dataset()
        keep = {
            "all": lambda card: True,
            "review": lambda card: bool(card.review),
            "meaningful": lambda card: card.meaningful,
            "not_meaningful": lambda card: not card.meaningful,
        }.get(show)
        if keep is None:
            raise HTTPException(422, "show must be all, review, meaningful or not_meaningful")
        chosen = [card for card in d.select(campaign, officer) if keep(card)]
        return [row(card, d.calls[card.call_id]) for card in reversed(chosen)]

    @app.get("/api/calls/{call_id}")
    def call_detail(call_id: str) -> dict:
        d = dataset()
        card = d.cards.get(call_id)
        if card is None:
            raise HTTPException(404, "No such call")
        call = d.calls[call_id]
        labels = {step.id: step.label for step in d.rubric.steps}
        labels |= {flag.id: flag.label for flag in d.rubric.flags}
        evidence = {item.id: item for item in card.evidence}
        covered = {step for step, done in call.steps.items() if done}
        return {
            **row(card, call),
            "confidence": card.confidence,
            "steps": [
                {
                    "id": step,
                    "label": labels[step],
                    "verdict": verdict,
                    "quote": evidence[step].quote if step in evidence else "",
                    "line": evidence[step].line if step in evidence else None,
                    "verified": evidence[step].verified if step in evidence else None,
                }
                for step, verdict in card.steps.items()
            ],
            "flag_details": [
                {"id": f.id, "label": labels[f.id], "quote": f.quote, "line": f.line}
                for f in card.flags
            ],
            "transcript": d.transcripts.get(call_id, []),
            "audio": f"/audio/{call_id}.mp3" if (d.audio / f"{call_id}.mp3").exists() else None,
            # Known only because the calls are synthetic. Shown on request, for checking.
            "truth": {
                "outcome": call.outcome,
                "covered": sorted(covered),
                "flags": call.flags,
                "meaningful": d.rubric.is_meaningful(call.outcome, covered),
            },
        }

    @app.get("/audio/{name}")
    def recording(name: str) -> FileResponse:
        d = dataset()
        call_id = name.removesuffix(".mp3")
        path = d.audio / f"{call_id}.mp3"
        if call_id not in d.calls or not path.exists():
            raise HTTPException(404, "No recording on disk. Run `callqa synthesize`.")
        return FileResponse(path, media_type="audio/mpeg")

    return app


app = create_app()
