"""Where each stage keeps its output. Every stage reads the previous one's file.

There are two sets of calls, built the same way from different seeds. The development set
is the one the grader's prompt and the masking rules were worked out on. The held-out set
was generated after those were frozen, and its results are the ones to quote.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from callqa.rubric import ROOT

SEEDS = {"dev": 11, "heldout": 23}
DEFAULT_SET = os.environ.get("CALLQA_SET", "heldout")
MODEL = ROOT / "models" / "ggml-base.en.bin"
SPEND = ROOT / "results" / "spend.json"


@dataclass(frozen=True)
class Paths:
    """The files of one set of calls."""

    name: str
    root: Path = ROOT

    @property
    def calls(self) -> Path:  # the synthetic calls and their true labels
        return self.root / "data" / self.name / "calls.jsonl"

    @property
    def transcripts(self) -> Path:  # speech-recognition output, before masking
        return self.root / "data" / self.name / "transcripts.jsonl"

    @property
    def audio(self) -> Path:  # not committed: regenerate with `callqa synthesize`
        return self.root / "data" / self.name / "audio"

    @property
    def results(self) -> Path:
        return self.root / "results" / self.name

    def scorecards(self, source: str, model: str) -> Path:
        """One file per grading run, named for the transcript source and the model."""
        return self.results / f"scorecards.{source}.{model.removeprefix('claude-')}.jsonl"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
