import math
import struct

from callqa import audio
from callqa.synth import Call, Turn


def tone(seconds: float) -> bytes:
    count = int(seconds * audio.RATE)
    return b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / audio.RATE)))
        for i in range(count)
    )


def quiet(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * audio.RATE)


def make_call(outcome: str = "conversation") -> Call:
    return Call(
        call_id="call_x",
        officer="A",
        officer_voice="v1",
        customer="B",
        customer_voice="v2",
        campaign="cold",
        started_at="2026-11-02T09:00",
        outcome=outcome,
        steps={},
        flags=[],
        personal=[],
        turns=[Turn("customer", "Hello?"), Turn("agent", "Hi."), Turn("customer", "Yes?")],
    )


def test_assemble_keeps_one_side_silent_while_the_other_speaks():
    agent, customer = audio.assemble(make_call(), synth=lambda voice, text: tone(0.5))
    assert len(agent) == len(customer)
    spoken = {
        "agent": [(s / audio.RATE, e / audio.RATE) for s, e in audio.utterances(agent)],
        "customer": [(s / audio.RATE, e / audio.RATE) for s, e in audio.utterances(customer)],
    }
    assert len(spoken["agent"]) == 1 and len(spoken["customer"]) == 2
    (first_start, first_end), (last_start, _) = spoken["customer"]
    agent_start, agent_end = spoken["agent"][0]
    assert first_start < first_end <= agent_start + 2 * audio.PAD_S
    assert agent_end <= last_start + 2 * audio.PAD_S


def test_assemble_uses_each_speakers_voice():
    used = []
    audio.assemble(make_call(), synth=lambda voice, text: used.append(voice) or tone(0.1))
    assert used == ["v2", "v1", "v2"]


def test_voicemail_greeting_is_followed_by_a_tone():
    plain, _ = audio.assemble(make_call(), synth=lambda voice, text: tone(0.2))
    beeped, _ = audio.assemble(make_call("voicemail"), synth=lambda voice, text: tone(0.2))
    assert len(beeped) > len(plain)


def test_utterances_split_on_silence_and_join_short_pauses():
    track = quiet(0.5) + tone(0.4) + quiet(0.1) + tone(0.4) + quiet(0.6) + tone(0.3) + quiet(0.5)
    spans = [(s / audio.RATE, e / audio.RATE) for s, e in audio.utterances(track)]
    assert len(spans) == 2  # the 0.1 s pause is inside one utterance, the 0.6 s one is not
    assert abs(spans[0][0] - 0.4) < 0.05 and abs(spans[0][1] - 1.5) < 0.05
    assert abs(spans[1][0] - 1.9) < 0.05 and abs(spans[1][1] - 2.4) < 0.05


def test_utterances_of_silence_is_empty():
    assert audio.utterances(quiet(1.0)) == []


def test_stereo_round_trip(tmp_path):
    left, right = tone(0.2), quiet(0.2)
    path = tmp_path / "call.wav"
    audio.write_wav(path, left, right)
    assert audio.split_stereo(path) == (left, right)


def test_segments_are_placed_on_the_call_timeline_without_noise_markers():
    result = {
        "transcription": [
            {"offsets": {"from": 0, "to": 900}, "text": " Hello there."},
            {"offsets": {"from": 900, "to": 1500}, "text": " [BLANK_AUDIO]"},
            {"offsets": {"from": 1500, "to": 2000}, "text": " (beep) Bye."},
        ]
    }
    segments = audio.segments_from(result, "agent", offset=10.0)
    assert [(s.start, s.end, s.text) for s in segments] == [
        (10.0, 10.9, "Hello there."),
        (11.5, 12.0, "Bye."),
    ]


def test_merge_orders_by_time_and_joins_one_speakers_pieces():
    pieces = [
        audio.Segment("customer", 5.0, 6.0, "Yes."),
        audio.Segment("agent", 0.0, 2.0, "Hi."),
        audio.Segment("agent", 2.1, 4.0, "This is Maria."),
        audio.Segment("agent", 7.0, 8.0, "Great."),
    ]
    merged = audio.merge(pieces)
    assert [(s.speaker, s.start, s.end, s.text) for s in merged] == [
        ("agent", 0.0, 4.0, "Hi. This is Maria."),
        ("customer", 5.0, 6.0, "Yes."),
        ("agent", 7.0, 8.0, "Great."),
    ]


def test_transcribe_cuts_channels_and_offsets_each_piece(tmp_path, monkeypatch):
    agent = quiet(1.0) + tone(0.5) + quiet(2.0)
    customer = tone(0.5) + quiet(1.5) + tone(0.5) + quiet(1.0)
    recording = tmp_path / "call_x.wav"
    audio.write_wav(recording, agent, customer)

    def fake_whisper(wavs, model):
        assert all(wav.exists() for wav in wavs)
        return [
            {"transcription": [{"offsets": {"from": 0, "to": 400}, "text": f" {wav.name}"}]}
            for wav in wavs
        ]

    monkeypatch.setattr(audio, "run_whisper", fake_whisper)
    segments = audio.transcribe(recording, tmp_path / "model.bin", tmp_path)
    assert [s.speaker for s in segments] == ["customer", "agent", "customer"]
    assert [round(s.start, 1) for s in segments] == [0.0, 0.9, 1.9]
    assert list(tmp_path.glob("call_x.*.wav")) == []  # the pieces are cleaned up
