"""Audio: synthesise a two-channel recording of a call, then transcribe it per channel.

Phone systems record each side of a call on its own channel. That makes "who said what" a
matter of which channel the words came from, with no speaker-diarisation model needed. The
recordings here are built the same way: the loan officer on the left, the contact on the
right, both on one timeline.

Speech comes from the macOS ``say`` command and transcription from whisper.cpp, so this
stage runs only on a Mac with ``whisper-cli`` installed. Everything after it runs anywhere.
"""

import json
import math
import re
import struct
import subprocess
import wave
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from callqa.synth import Call

RATE = 16_000  # what whisper.cpp expects
GAP_S = 0.35  # pause between turns
FRAME = RATE // 50  # 20 ms, the unit used to find speech
LOUD = 250  # a frame whose peak is above this (of 32767) counts as speech
JOIN_S = 0.25  # speech separated by less silence than this is one utterance
PAD_S = 0.1  # kept on each side of an utterance so no word is clipped
_NOISE = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # [BLANK_AUDIO], (beep) and similar

Synth = Callable[[str, str], bytes]  # (voice, text) -> 16-bit mono PCM at RATE


@dataclass
class Segment:
    speaker: str  # "agent" or "customer"
    start: float  # seconds from the start of the call
    end: float
    text: str


def say(voice: str, text: str) -> bytes:
    """Speak ``text`` with a macOS voice and return the raw samples."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "line.wav"
        command = ["say", "-v", voice, "--file-format=WAVE", f"--data-format=LEI16@{RATE}"]
        subprocess.run([*command, "-o", str(path), text], check=True)
        with wave.open(str(path)) as clip:
            return clip.readframes(clip.getnframes())


def _silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * RATE)


def _beep(seconds: float = 0.4, hz: float = 1000.0) -> bytes:
    count = int(seconds * RATE)
    return b"".join(
        struct.pack("<h", int(9000 * math.sin(2 * math.pi * hz * i / RATE))) for i in range(count)
    )


def assemble(call: Call, synth: Synth = say) -> tuple[bytes, bytes]:
    """Lay the turns end to end on two tracks. While one side speaks, the other is silent."""
    agent, customer = bytearray(), bytearray()
    for index, turn in enumerate(call.turns):
        voice = call.officer_voice if turn.speaker == "agent" else call.customer_voice
        clip = synth(voice, turn.text)
        if call.outcome == "voicemail" and index == 0:
            clip += _silence(0.2) + _beep()  # the tone after the greeting
        quiet = _silence(len(clip) / 2 / RATE)
        gap = _silence(GAP_S)
        speaking, listening = (agent, customer) if turn.speaker == "agent" else (customer, agent)
        speaking += clip + gap
        listening += quiet + gap
    return bytes(agent), bytes(customer)


def write_wav(path: Path, *channels: bytes) -> None:
    """Write one or two 16-bit tracks as a mono or stereo WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(channels) == 1:
        frames = channels[0]
    else:
        left, right = channels
        frames = b"".join(
            left[i : i + 2] + right[i : i + 2] for i in range(0, min(len(left), len(right)), 2)
        )
    with wave.open(str(path), "wb") as out:
        out.setnchannels(len(channels))
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(frames)


def split_stereo(path: Path) -> tuple[bytes, bytes]:
    with wave.open(str(path)) as recording:
        frames = recording.readframes(recording.getnframes())
    left = b"".join(frames[i : i + 2] for i in range(0, len(frames), 4))
    right = b"".join(frames[i + 2 : i + 4] for i in range(0, len(frames), 4))
    return left, right


def utterances(samples: bytes) -> list[tuple[int, int]]:
    """Where one channel has speech, as (first sample, last sample) pairs.

    On a two-channel recording each side is quiet while the other talks, so a plain loudness
    threshold finds the turns. Transcribing a whole channel in one go does not work well:
    Whisper runs the short replies together and loses their timing.
    """
    pcm = array("h", samples)
    loud = [max(map(abs, pcm[i : i + FRAME])) > LOUD for i in range(0, len(pcm) - FRAME + 1, FRAME)]
    join, pad = int(JOIN_S * RATE / FRAME), int(PAD_S * RATE)
    spans: list[list[int]] = []
    for index, speech in enumerate(loud):
        if not speech:
            continue
        if spans and index - spans[-1][1] <= join:
            spans[-1][1] = index
        else:
            spans.append([index, index])
    return [
        (max(0, first * FRAME - pad), min(len(pcm), (last + 1) * FRAME + pad))
        for first, last in spans
    ]


def run_whisper(wavs: list[Path], model: Path) -> list[dict]:
    """Transcribe mono files with whisper.cpp in one process; one JSON result per file."""
    if not wavs:
        return []
    subprocess.run(
        ["whisper-cli", "-m", str(model), "-oj", "-np", *map(str, wavs)],
        check=True,
        capture_output=True,
    )
    results = []
    for wav in wavs:
        output = Path(f"{wav}.json")  # whisper appends the extension to the input path
        results.append(json.loads(output.read_text()))
        output.unlink()
    return results


def segments_from(whisper_json: dict, speaker: str, offset: float = 0.0) -> list[Segment]:
    """Whisper's segments for one utterance, placed on the call's timeline.

    ``offset`` is where the utterance starts in the call, in seconds. The noise markers
    Whisper emits for tones and silence are dropped.
    """
    found = []
    for item in whisper_json.get("transcription", []):
        text = " ".join(_NOISE.sub(" ", item["text"]).split())
        if text:
            start, end = (item["offsets"][key] / 1000 + offset for key in ("from", "to"))
            found.append(Segment(speaker, round(start, 2), round(end, 2), text))
    return found


def merge(pieces: list[Segment]) -> list[Segment]:
    """Put both channels in time order, joining consecutive pieces from one speaker."""
    merged: list[Segment] = []
    for segment in sorted(pieces, key=lambda s: (s.start, s.speaker)):
        last = merged[-1] if merged else None
        if last and last.speaker == segment.speaker:
            merged[-1] = Segment(
                last.speaker, last.start, segment.end, f"{last.text} {segment.text}"
            )
        else:
            merged.append(segment)
    return merged


def transcribe(recording: Path, model: Path, scratch: Path) -> list[Segment]:
    """A stereo call recording to a speaker-labelled transcript."""
    left, right = split_stereo(recording)
    clips: list[tuple[Path, str, float]] = []  # file, speaker, start in seconds
    for speaker, samples in (("agent", left), ("customer", right)):
        for number, (first, last) in enumerate(utterances(samples)):
            clip = scratch / f"{recording.stem}.{speaker}.{number:03d}.wav"
            write_wav(clip, samples[first * 2 : last * 2])
            clips.append((clip, speaker, first / RATE))
    try:
        results = run_whisper([clip for clip, _, _ in clips], model)
    finally:
        for clip, _, _ in clips:
            clip.unlink(missing_ok=True)
    pieces = [
        segment
        for (_, speaker, start), result in zip(clips, results, strict=True)
        for segment in segments_from(result, speaker, start)
    ]
    return merge(pieces)


def reference_segments(call: Call) -> list[Segment]:
    """The transcript a perfect transcriber would produce: the original lines, no timings."""
    return [
        Segment(turn.speaker, float(i), float(i), turn.text) for i, turn in enumerate(call.turns)
    ]


def to_json(segments: list[Segment]) -> list[dict]:
    return [asdict(segment) for segment in segments]


def from_json(rows: list[dict]) -> list[Segment]:
    return [Segment(**row) for row in rows]


def compress(wav: Path, out: Path) -> None:
    """A small stereo MP3 for the dashboard's player."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "48k", str(out)], check=True
    )
