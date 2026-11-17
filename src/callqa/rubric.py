"""The call script as data, and the business rule for a meaningful call."""

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "config" / "script.yml"


@dataclass(frozen=True)
class Step:
    id: str
    label: str
    description: str
    campaigns: tuple[str, ...]


@dataclass(frozen=True)
class Flag:
    id: str
    label: str
    description: str


@dataclass(frozen=True)
class Rubric:
    company: str
    outcomes: dict[str, str]
    steps: tuple[Step, ...]
    flags: tuple[Flag, ...]
    meaningful_outcome: str
    meaningful_require: tuple[str, ...]
    meaningful_count: int
    meaningful_of: tuple[str, ...]

    @classmethod
    def load(cls, path: Path = SCRIPT_PATH) -> "Rubric":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        def clean(text: str) -> str:
            return " ".join(text.split())

        rule = raw["meaningful"]
        return cls(
            company=raw["company"],
            outcomes={key: clean(value) for key, value in raw["outcomes"].items()},
            steps=tuple(
                Step(s["id"], s["label"], clean(s["description"]), tuple(s["campaigns"]))
                for s in raw["steps"]
            ),
            flags=tuple(Flag(f["id"], f["label"], clean(f["description"])) for f in raw["flags"]),
            meaningful_outcome=rule["outcome"],
            meaningful_require=tuple(rule["require"]),
            meaningful_count=rule["at_least"]["count"],
            meaningful_of=tuple(rule["at_least"]["of"]),
        )

    def steps_for(self, campaign: str) -> list[Step]:
        return [step for step in self.steps if campaign in step.campaigns]

    def is_meaningful(self, outcome: str, covered: set[str]) -> bool:
        """The business rule, applied the same way to model verdicts and to true labels."""
        return (
            outcome == self.meaningful_outcome
            and all(step in covered for step in self.meaningful_require)
            and sum(step in covered for step in self.meaningful_of) >= self.meaningful_count
        )

    def adherence(self, campaign: str, covered: set[str]) -> float:
        """Share of the campaign's script steps that were covered."""
        steps = self.steps_for(campaign)
        return sum(step.id in covered for step in steps) / len(steps)
