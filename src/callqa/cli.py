"""The pipeline, one command per stage.

callqa generate      build the synthetic calls and their true labels (free)
callqa synthesize    speak each call into a two-channel recording (macOS)
callqa transcribe    transcribe each channel with whisper.cpp and mask personal details
callqa grade         grade every call against the script (paid)
callqa evaluate      compare the grades with the true labels (free)
callqa serve         the dashboard
"""

import argparse
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from callqa import audio, store
from callqa.redact import redact
from callqa.rubric import Rubric
from callqa.synth import Call, generate

DEFAULT_MODEL = os.environ.get("CALLQA_MODEL", "claude-haiku-5-5")


def load_calls() -> list[Call]:
    return [Call.from_json(row) for row in store.read(store.CALLS)]


def cmd_generate(args: argparse.Namespace) -> None:
    calls = generate(count=args.count, seed=args.seed)
    store.write(store.CALLS, [call.to_json() for call in calls])
    outcomes: dict[str, int] = {}
    for call in calls:
        outcomes[call.outcome] = outcomes.get(call.outcome, 0) + 1
    print(f"{len(calls)} calls written to {store.CALLS.relative_to(store.ROOT)}: {outcomes}")


def cmd_synthesize(args: argparse.Namespace) -> None:
    calls = load_calls()[: args.limit]

    def one(call: Call) -> float:
        wav = store.AUDIO / f"{call.call_id}.wav"
        if not wav.exists() or args.force:
            agent, customer = audio.assemble(call)
            audio.write_wav(wav, agent, customer)
            audio.compress(wav, wav.with_suffix(".mp3"))
        return wav.stat().st_size / (audio.RATE * 4)  # seconds of stereo 16-bit audio

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        seconds = list(pool.map(one, calls))
    print(f"{len(calls)} recordings, {sum(seconds) / 60:.1f} minutes of audio in total.")


def cmd_transcribe(args: argparse.Namespace) -> None:
    if not store.MODEL.exists():
        raise SystemExit(f"Speech model not found at {store.MODEL}. See the README.")
    calls = load_calls()[: args.limit]
    scratch = store.ROOT / ".cache" / "whisper"
    scratch.mkdir(parents=True, exist_ok=True)

    def one(call: Call) -> dict:
        segments = audio.transcribe(store.AUDIO / f"{call.call_id}.wav", store.MODEL, scratch)
        masked, counts = [], {}
        for segment in segments:
            text, found = redact(segment.text)
            masked.append(audio.Segment(segment.speaker, segment.start, segment.end, text))
            for label, count in found.items():
                counts[label] = counts.get(label, 0) + count
        return {
            "call_id": call.call_id,
            "segments": audio.to_json(segments),
            "masked": audio.to_json(masked),
            "masked_counts": counts,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, calls))
    store.write(store.TRANSCRIPTS, rows)
    print(f"{len(rows)} transcripts written to {store.TRANSCRIPTS.relative_to(store.ROOT)}.")


def cmd_grade(args: argparse.Namespace) -> None:
    from callqa.grader import grade
    from callqa.llm import LLM

    rubric = Rubric.load()
    calls = load_calls()[: args.limit]
    if args.source == "asr":
        transcripts = {row["call_id"]: row for row in store.read(store.TRANSCRIPTS)}
        inputs = {c.call_id: audio.from_json(transcripts[c.call_id]["masked"]) for c in calls}
    else:  # the original lines, as if transcription were perfect
        inputs = {}
        for call in calls:
            perfect = audio.reference_segments(call)
            inputs[call.call_id] = [
                audio.Segment(s.speaker, s.start, s.end, redact(s.text)[0]) for s in perfect
            ]
    llm = LLM(
        cache_dir=store.ROOT / ".cache" / "llm", ledger_path=store.SPEND, budget_usd=args.budget
    )

    async def run() -> list:
        return await asyncio.gather(
            *(
                grade(llm, args.model, rubric, call.call_id, call.campaign, inputs[call.call_id])
                for call in calls
            ),
            return_exceptions=True,
        )

    results = asyncio.run(run())
    cards = [card for card in results if not isinstance(card, BaseException)]
    for call, result in zip(calls, results, strict=True):
        if isinstance(result, BaseException):  # reported, never guessed at
            print(f"{call.call_id} was not graded: {type(result).__name__}: {result}")
    out = store.scorecards_path(args.source, args.model)
    store.write(out, [card.to_json() for card in cards])
    print(f"{len(cards)} of {len(calls)} calls graded, in {out.relative_to(store.ROOT)}.")
    print(f"Total spend so far: ${llm.ledger.total:.4f}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from callqa import report

    text = report.render()
    if args.write:
        (store.RESULTS / "report.md").write_text(text + "\n")
    print(text)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("callqa.api:app", host="0.0.0.0", port=args.port)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(prog="callqa", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    make = commands.add_parser("generate", help="build the synthetic calls (free)")
    make.add_argument("--count", type=int, default=160)
    make.add_argument("--seed", type=int, default=11)
    make.set_defaults(func=cmd_generate)

    for name, func, text in (
        ("synthesize", cmd_synthesize, "speak each call into a recording (macOS)"),
        ("transcribe", cmd_transcribe, "transcribe and mask every recording"),
    ):
        stage = commands.add_parser(name, help=text)
        stage.add_argument("--limit", type=int, default=None)
        stage.add_argument("--workers", type=int, default=4)
        stage.add_argument("--force", action="store_true")
        stage.set_defaults(func=func)

    grading = commands.add_parser("grade", help="grade every call against the script (paid)")
    grading.add_argument("--source", choices=["asr", "clean"], default="asr")
    grading.add_argument("--model", default=DEFAULT_MODEL)
    grading.add_argument("--budget", type=float, default=5.0)
    grading.add_argument("--limit", type=int, default=None)
    grading.set_defaults(func=cmd_grade)

    scoring = commands.add_parser("evaluate", help="compare grades with the true labels (free)")
    scoring.add_argument("--write", action="store_true", help="also update results/report.md")
    scoring.set_defaults(func=cmd_evaluate)

    serving = commands.add_parser("serve", help="run the dashboard")
    serving.add_argument("--port", type=int, default=8020)
    serving.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
