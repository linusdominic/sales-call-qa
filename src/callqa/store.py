"""Where each stage keeps its output. Every stage reads the previous one's file."""

import json
from pathlib import Path

from callqa.rubric import ROOT

DATA = ROOT / "data"
RESULTS = ROOT / "results"
AUDIO = DATA / "audio"  # not committed: regenerate with `callqa synthesize`
MODEL = ROOT / "models" / "ggml-base.en.bin"

CALLS = DATA / "calls.jsonl"  # the synthetic calls and their true labels
TRANSCRIPTS = DATA / "transcripts.jsonl"  # speech-recognition output, raw and masked
SPEND = RESULTS / "spend.json"


def scorecards_path(source: str, model: str) -> Path:
    """One file per grading run, named for the transcript source and the model."""
    return RESULTS / f"scorecards.{source}.{model.removeprefix('claude-')}.jsonl"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
