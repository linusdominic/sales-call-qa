"""Compare every grading run with the true labels and write the report.

The calls are synthetic, so the right answer for each one is known exactly: how it ended,
which script steps the loan officer covered, which rules they broke. Everything here is
counted from the files the pipeline stages wrote. No model is called.
"""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from callqa import audio, store
from callqa.grader import Scorecard
from callqa.redact import leaked, redact
from callqa.rubric import Rubric
from callqa.synth import SKIPPED, TRAPS, Call

# Lines planted to look like something they are not. Keyed by what a grader might get wrong.
TRAP_LINES = {
    "Officer declines to guarantee approval (must not raise guaranteed_approval)": (
        "flag",
        "guaranteed_approval",
        TRAPS[0][1],
    ),
    "Officer declines to promise a rate (must not raise rate_promise)": (
        "flag",
        "rate_promise",
        TRAPS[1][1],
    ),
    'Officer says "no rush" (must not raise high_pressure)': ("flag", "high_pressure", TRAPS[2][1]),
    'Officer says "licensed" with no number (nmls_id must be missing)': (
        "step",
        "nmls_id",
        SKIPPED["nmls_id"][1][0],
    ),
    "Contact asks about recording, officer dodges (recording_disclosure must be missing)": (
        "step",
        "recording_disclosure",
        SKIPPED["recording_disclosure"][1][1],
    ),
    'Officer says "reach out if you want" (next_step must be missing)': (
        "step",
        "next_step",
        SKIPPED["next_step"][0][0],
    ),
}

_NUMBER_WORDS = {
    *("zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"),
    *("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen"),
    *("eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy"),
    *("eighty", "ninety", "hundred", "thousand", "point", "percent", "dollars", "nmls", "id"),
}


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    scale = 1 + z * z / total
    return (max(0.0, (centre - spread) / scale), min(1.0, (centre + spread) / scale))


def pct(hits: int, total: int) -> str:
    return f"{100 * hits / total:.1f}% ({hits}/{total})" if total else "n/a"


def pct_ci(hits: int, total: int) -> str:
    low, high = wilson(hits, total)
    return f"{pct(hits, total)}, 95% CI {100 * low:.0f} to {100 * high:.0f}%" if total else "n/a"


def words(text: str) -> list[str]:
    """Words for the error-rate comparison, without numbers.

    The original lines spell numbers the way they are spoken ("5 5 5, 0 1 6 1", "seven
    twenty") and Whisper writes digits ("555-0161", "720"). Counting those as errors would
    measure formatting, so digits, number words and single letters are left out.
    """
    flat = re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("'", "").replace("’", ""))
    return [
        word
        for word in flat.split()
        if len(word) > 1 and not any(c.isdigit() for c in word) and word not in _NUMBER_WORDS
    ]


def edit_distance(reference: list[str], heard: list[str]) -> int:
    row = list(range(len(heard) + 1))
    for i, expected in enumerate(reference, 1):
        previous, row[0] = row[0], i
        for j, got in enumerate(heard, 1):
            previous, row[j] = (
                row[j],
                min(row[j] + 1, row[j - 1] + 1, previous + (expected != got)),
            )
    return row[-1]


def speaker_runs(speakers: list[str]) -> list[str]:
    return [s for i, s in enumerate(speakers) if i == 0 or s != speakers[i - 1]]


@dataclass
class Truth:
    outcome: str
    covered: set[str]
    applicable: list[str]  # the steps that are judged: empty unless a conversation
    flags: set[str]
    meaningful: bool
    adherence: float | None


def truth_for(call: Call, rubric: Rubric) -> Truth:
    covered = {step for step, done in call.steps.items() if done}
    conversation = call.outcome == rubric.meaningful_outcome
    return Truth(
        outcome=call.outcome,
        covered=covered,
        applicable=list(call.steps),
        flags=set(call.flags),
        meaningful=rubric.is_meaningful(call.outcome, covered),
        adherence=rubric.adherence(call.campaign, covered) if conversation else None,
    )


@dataclass
class Run:
    """One grading run compared with the truth."""

    source: str
    model: str
    cards: dict[str, Scorecard]
    calls: int = 0
    outcome_ok: int = 0
    confusions: dict[tuple[str, str], int] = field(default_factory=dict)
    meaningful: dict[str, int] = field(default_factory=lambda: dict(tp=0, fp=0, fn=0, tn=0))
    step_total: int = 0
    step_ok: int = 0
    by_step: dict[str, dict[str, int]] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=lambda: dict(tp=0, fp=0, fn=0))
    evidence_total: int = 0
    evidence_ok: int = 0
    wrong: dict[str, list[str]] = field(default_factory=dict)  # call id -> what was wrong
    queued: set[str] = field(default_factory=set)
    true_flag_calls: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        source = "Whisper transcript" if self.source == "asr" else "original text"
        return f"{self.model}, {source}"

    @property
    def cost(self) -> float:
        return sum(card.cost_usd for card in self.cards.values())

    @property
    def latencies(self) -> list[float]:
        return sorted(card.latency_s for card in self.cards.values())


def compare(source: str, model: str, cards: list[Scorecard], calls: list[Call], rubric) -> Run:
    run = Run(source, model, {card.call_id: card for card in cards})
    for call in calls:
        card = run.cards.get(call.call_id)
        if card is None:
            continue
        truth = truth_for(call, rubric)
        problems: list[str] = []
        run.calls += 1

        if card.outcome == truth.outcome:
            run.outcome_ok += 1
        else:
            pair = (truth.outcome, card.outcome)
            run.confusions[pair] = run.confusions.get(pair, 0) + 1
            problems.append(f"outcome {truth.outcome} graded as {card.outcome}")

        key = ("t" if card.meaningful == truth.meaningful else "f") + (
            "p" if card.meaningful else "n"
        )
        run.meaningful[key] += 1
        if card.meaningful != truth.meaningful:
            problems.append(
                "graded meaningful, was not" if card.meaningful else "graded not meaningful, was"
            )

        for step in truth.applicable:
            was, graded = step in truth.covered, step in card.covered
            tally = run.by_step.setdefault(step, dict(total=0, ok=0, false_covered=0, false_miss=0))
            tally["total"] += 1
            run.step_total += 1
            if was == graded:
                tally["ok"] += 1
                run.step_ok += 1
            else:
                tally["false_covered" if graded else "false_miss"] += 1
                problems.append(f"{step} graded {'covered' if graded else 'missing'}")

        graded_flags = {flag.id for flag in card.flags}
        run.flags["tp"] += len(graded_flags & truth.flags)
        run.flags["fp"] += len(graded_flags - truth.flags)
        run.flags["fn"] += len(truth.flags - graded_flags)
        problems += [f"false flag {flag}" for flag in sorted(graded_flags - truth.flags)]
        problems += [f"missed flag {flag}" for flag in sorted(truth.flags - graded_flags)]
        if truth.flags:
            run.true_flag_calls.add(call.call_id)

        for item in [*card.evidence, *card.flags]:
            run.evidence_total += 1
            run.evidence_ok += item.verified
        if card.review:
            run.queued.add(call.call_id)
        if problems:
            run.wrong[call.call_id] = problems
    return run


def load_runs(calls: list[Call], rubric: Rubric, results: Path = store.RESULTS) -> list[Run]:
    runs = []
    for path in sorted(results.glob("scorecards.*.jsonl")):
        _, source, model = path.stem.split(".", 2)
        cards = [Scorecard.from_json(row) for row in store.read(path)]
        runs.append(compare(source, f"claude-{model}", cards, calls, rubric))
    # The run that matters first: the real pipeline (speech recognition), cheapest model.
    return sorted(runs, key=lambda run: (run.source != "asr", "haiku" not in run.model, run.model))


def ratio(top: int, bottom: int) -> str:
    return f"{100 * top / bottom:.1f}%" if bottom else "n/a"


def transcription_section(calls: list[Call], transcripts: dict[str, dict]) -> list[str]:
    errors = reference_words = order_ok = 0
    seconds = 0.0
    for call in calls:
        row = transcripts.get(call.call_id)
        if row is None:
            continue
        heard = audio.from_json(row["segments"])
        seconds += max((segment.end for segment in heard), default=0.0)
        expected = words(" ".join(turn.text for turn in call.turns))
        errors += edit_distance(expected, words(" ".join(segment.text for segment in heard)))
        reference_words += len(expected)
        order_ok += speaker_runs([s.speaker for s in heard]) == speaker_runs(
            [turn.speaker for turn in call.turns]
        )
    done = sum(call.call_id in transcripts for call in calls)
    return [
        "## Transcription",
        "",
        f"{done} two-channel recordings, {seconds / 60:.0f} minutes of audio, transcribed with "
        "whisper.cpp `base.en` one channel at a time.",
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Word error rate, numbers excluded | {ratio(errors, reference_words)} "
        f"({errors} errors in {reference_words} words) |",
        f"| Calls whose speaker order matches the original | {pct(order_ok, done)} |",
        "",
    ]


def masking_section(calls: list[Call], transcripts: dict[str, dict]) -> list[str]:
    kinds = {"phone": "Phone number", "ssn": "Social Security number"}
    kinds |= {"email": "Email address", "dob": "Date of birth"}
    tally = {kind: dict(planted=0, asr=0, clean=0) for kind in kinds}
    for call in calls:
        row = transcripts.get(call.call_id)
        if row is None:
            continue
        masked_asr = " ".join(segment["text"] for segment in row["masked"])
        masked_clean = " ".join(redact(turn.text)[0] for turn in call.turns)
        for item in call.personal:
            tally[item["kind"]]["planted"] += 1
            tally[item["kind"]]["asr"] += not leaked(item["kind"], item["value"], masked_asr)
            tally[item["kind"]]["clean"] += not leaked(item["kind"], item["value"], masked_clean)
    lines = [
        "## Masking of personal details",
        "",
        "Details the contacts say aloud, and how many were unreadable after masking.",
        "",
        "| Detail | Planted | Masked in Whisper transcript | Masked in original text |",
        "|---|---|---|---|",
    ]
    for kind, label in kinds.items():
        t = tally[kind]
        lines.append(
            f"| {label} | {t['planted']} | {pct(t['asr'], t['planted'])} "
            f"| {pct(t['clean'], t['planted'])} |"
        )
    total = {key: sum(t[key] for t in tally.values()) for key in ("planted", "asr", "clean")}
    lines.append(
        f"| All | {total['planted']} | {pct(total['asr'], total['planted'])} "
        f"| {pct(total['clean'], total['planted'])} |"
    )
    return [*lines, ""]


def grading_section(runs: list[Run]) -> list[str]:
    lines = [
        "## Grading against the true labels",
        "",
        "| Run | Outcome | Meaningful: precision | Meaningful: recall | Script steps "
        "| Flags: precision | Flags: recall | Calls with no error | Cost | Median time |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        m, f = run.meaningful, run.flags
        lines.append(
            f"| {run.name} | {pct(run.outcome_ok, run.calls)} "
            f"| {pct(m['tp'], m['tp'] + m['fp'])} | {pct(m['tp'], m['tp'] + m['fn'])} "
            f"| {pct(run.step_ok, run.step_total)} "
            f"| {pct(f['tp'], f['tp'] + f['fp'])} | {pct(f['tp'], f['tp'] + f['fn'])} "
            f"| {pct(run.calls - len(run.wrong), run.calls)} "
            f"| ${run.cost:.2f} | {median(run.latencies):.1f} s |"
        )
    lines += [
        "",
        "Script steps are counted over the calls that really were conversations: one decision "
        "per step per call. Flags are counted over every call.",
        "",
    ]
    return lines


def detail_section(run: Run, calls: list[Call], rubric: Rubric) -> list[str]:
    by_id = {call.call_id: call for call in calls}
    m = run.meaningful
    lines = [
        f"## In detail: {run.name}",
        "",
        f"- Outcome: {pct_ci(run.outcome_ok, run.calls)}.",
        f"- Meaningful or not: {pct_ci(m['tp'] + m['tn'], run.calls)}. "
        f"{m['tp']} true positives, {m['fp']} false positives, {m['fn']} false negatives, "
        f"{m['tn']} true negatives.",
        f"- Script step decisions: {pct_ci(run.step_ok, run.step_total)}.",
        f"- Quoted evidence found in the transcript: {pct(run.evidence_ok, run.evidence_total)}.",
        f"- Cost: ${run.cost:.4f} for {run.calls} calls, ${1000 * run.cost / run.calls:.2f} per "
        f"1,000 calls. Time per call: median {median(run.latencies):.1f} s, "
        f"slowest {run.latencies[-1]:.1f} s.",
        "",
        "### By script step",
        "",
        "| Step | Decisions | Correct | Graded covered, was missing "
        "| Graded missing, was covered |",
        "|---|---|---|---|---|",
    ]
    for step in rubric.steps:
        t = run.by_step.get(step.id)
        if t:
            lines.append(
                f"| {step.label} | {t['total']} | {pct(t['ok'], t['total'])} "
                f"| {t['false_covered']} | {t['false_miss']} |"
            )

    lines += ["", "### Trap lines", "", "| Trap | Calls | Handled correctly |", "|---|---|---|"]
    for label, (kind, target, text) in TRAP_LINES.items():
        marker = text.split("{")[0]
        hit = [
            call.call_id
            for call in calls
            if call.call_id in run.cards
            and any(turn.speaker == "agent" and marker in turn.text for turn in call.turns)
        ]
        if kind == "flag":
            good = sum(target not in {f.id for f in run.cards[c].flags} for c in hit)
        else:
            good = sum(target not in run.cards[c].covered for c in hit)
        lines.append(f"| {label} | {len(hit)} | {pct(good, len(hit))} |")

    wrong, queued = set(run.wrong), run.queued
    deserved = wrong | run.true_flag_calls
    lines += [
        "",
        "### Review queue",
        "",
        "A call goes to a person when it has a compliance flag, the model was not confident, or "
        "a quote could not be found in the transcript.",
        "",
        f"- Calls sent for review: {pct(len(queued), run.calls)}.",
        f"- Of those, calls with a real rule break or a grading error: "
        f"{pct(len(queued & deserved), len(queued))}.",
        f"- Calls with a grading error: {len(wrong)}. Caught by the queue: "
        f"{pct(len(wrong & queued), len(wrong))}.",
        f"- Calls with a real rule break: {len(run.true_flag_calls)}. In the queue: "
        f"{pct(len(run.true_flag_calls & queued), len(run.true_flag_calls))}.",
        "",
        "### What the dashboard shows per loan officer, against the truth",
        "",
        "| Loan officer | Calls | Meaningful: true | graded | Script adherence: true | graded "
        "| Flags: true | graded |",
        "|---|---|---|---|---|---|---|---|",
    ]
    worst = 0
    for officer in sorted({call.officer for call in calls}):
        mine = [c for c in calls if c.officer == officer and c.call_id in run.cards]
        truths = [truth_for(call, rubric) for call in mine]
        cards = [run.cards[call.call_id] for call in mine]
        true_meaningful = sum(t.meaningful for t in truths)
        graded_meaningful = sum(card.meaningful for card in cards)
        worst = max(worst, abs(true_meaningful - graded_meaningful))
        true_adherence = [t.adherence for t in truths if t.adherence is not None]
        graded_adherence = [card.adherence for card in cards if card.adherence is not None]
        lines.append(
            f"| {officer} | {len(mine)} | {true_meaningful} | {graded_meaningful} "
            f"| {100 * sum(true_adherence) / max(1, len(true_adherence)):.0f}% "
            f"| {100 * sum(graded_adherence) / max(1, len(graded_adherence)):.0f}% "
            f"| {sum(len(t.flags) for t in truths)} | {sum(len(card.flags) for card in cards)} |"
        )
    lines += [
        "",
        f"Largest gap in any officer's meaningful-call count: {worst}.",
        "",
        "### Every call the grader got wrong",
        "",
    ]
    if not run.wrong:
        lines.append("None.")
    for call_id in sorted(run.wrong):
        call = by_id[call_id]
        queue = "in review queue" if call_id in queued else "not in review queue"
        lines.append(
            f"- `{call_id}` ({call.officer}, {call.campaign}): "
            f"{'; '.join(run.wrong[call_id])}. {queue.capitalize()}."
        )
    return [*lines, ""]


def render(root: Path = store.ROOT) -> str:
    rubric = Rubric.load()
    calls = [Call.from_json(row) for row in store.read(root / "data" / "calls.jsonl")]
    transcript_path = root / "data" / "transcripts.jsonl"
    rows = store.read(transcript_path) if transcript_path.exists() else []
    transcripts = {row["call_id"]: row for row in rows}
    runs = load_runs(calls, rubric, root / "results")

    outcomes: dict[str, int] = {}
    for call in calls:
        outcomes[call.outcome] = outcomes.get(call.outcome, 0) + 1
    truths = [truth_for(call, rubric) for call in calls]
    mix = ", ".join(f"{count} {name}" for name, count in sorted(outcomes.items()))
    lines = [
        "# Evaluation report",
        "",
        "Built by `callqa evaluate` from the files in `data/` and `results/`. No model is called.",
        "",
        "## The calls",
        "",
        f"{len(calls)} synthetic calls by {len({call.officer for call in calls})} loan officers: "
        f"{mix}. By the rule in `config/script.yml`, "
        f"{sum(t.meaningful for t in truths)} are meaningful. "
        f"{sum(bool(t.flags) for t in truths)} contain at least one rule break.",
        "",
    ]
    if transcripts:
        lines += transcription_section(calls, transcripts)
        lines += masking_section(calls, transcripts)
    if runs:
        lines += grading_section(runs)
        lines += detail_section(runs[0], calls, rubric)
    else:
        lines += ["No grading runs yet. Run `callqa grade`.", ""]
    return "\n".join(lines).rstrip()
