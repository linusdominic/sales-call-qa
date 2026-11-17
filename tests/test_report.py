import copy

from conftest import masked_segments

from callqa import audio, report, store
from callqa.grader import Evidence
from callqa.synth import Turn


def test_words_drop_numbers_however_they_are_written():
    spoken = "My N M L S number is 6 0 3 2 8 8, about seven twenty."
    written = "My NMLS number is 603-288, about 720."
    assert report.words(spoken) == report.words(written) == ["my", "number", "is", "about"]


def test_edit_distance_counts_substitutions_insertions_and_deletions():
    reference = ["this", "is", "priya", "nair", "calling"]
    assert report.edit_distance(reference, reference) == 0
    assert report.edit_distance(reference, ["this", "is", "prianere", "calling"]) == 2
    assert report.edit_distance(reference, ["this", "is", "priya", "nair", "calling", "now"]) == 1
    assert report.edit_distance([], ["hello"]) == 1


def test_wilson_interval():
    low, high = report.wilson(95, 100)
    assert 0.88 < low < 0.90 and 0.97 < high < 0.99
    assert report.wilson(0, 0) == (0.0, 0.0)
    assert report.wilson(10, 10)[1] == 1.0


def test_speaker_runs_collapse_consecutive_turns():
    assert report.speaker_runs(["agent", "agent", "customer", "agent"]) == [
        "agent",
        "customer",
        "agent",
    ]


def test_perfect_grades_score_perfectly(calls, cards, rubric):
    run = report.compare("clean", "claude-haiku-5-5", cards, calls, rubric)
    assert run.calls == run.outcome_ok == len(calls)
    assert run.wrong == {} and run.confusions == {}
    assert run.step_ok == run.step_total > 0
    assert run.flags["fp"] == run.flags["fn"] == 0 and run.flags["tp"] > 0
    assert run.meaningful["fp"] == run.meaningful["fn"] == 0
    assert run.evidence_ok == run.evidence_total > 0
    assert run.queued == run.true_flag_calls  # only real rule breaks are queued


def test_each_kind_of_mistake_is_counted(calls, cards, rubric):
    cards = copy.deepcopy(cards)
    by_id = {card.call_id: card for card in cards}
    talk = [call for call in calls if call.outcome == "conversation"]

    wrong_outcome = next(call for call in calls if call.outcome == "voicemail")
    by_id[wrong_outcome.call_id].outcome = "wrong_number"

    missed = next(c for c in talk if c.steps["nmls_id"])
    by_id[missed.call_id].steps["nmls_id"] = "missing"

    invented = next(c for c in talk if not c.flags and c is not missed)
    by_id[invented.call_id].flags.append(Evidence("rate_promise", 0, "a quote", True))

    overlooked = next(c for c in talk if c.flags)
    by_id[overlooked.call_id].flags.clear()

    was = next(c for c in talk if by_id[c.call_id].meaningful and c is not missed)
    by_id[was.call_id].meaningful = False

    run = report.compare("asr", "claude-haiku-5-5", cards, calls, rubric)
    assert run.outcome_ok == len(calls) - 1
    assert run.confusions == {("voicemail", "wrong_number"): 1}
    assert run.by_step["nmls_id"]["false_miss"] == 1 and run.step_ok == run.step_total - 1
    assert run.flags["fp"] == 1 and run.flags["fn"] == len(overlooked.flags)
    assert run.meaningful["fn"] == 1
    assert "nmls_id graded missing" in run.wrong[missed.call_id]
    assert "false flag rate_promise" in run.wrong[invented.call_id]
    assert f"missed flag {overlooked.flags[0]}" in run.wrong[overlooked.call_id]
    assert "graded not meaningful, was" in run.wrong[was.call_id]


def write_run(root, calls, cards, source="asr", name="heldout", folder=""):
    paths = store.Paths(name, root)
    store.write(paths.calls, [call.to_json() for call in calls])
    rows = [
        {"call_id": call.call_id, "segments": audio.to_json(audio.reference_segments(call))}
        for call in calls
    ]
    store.write(paths.transcripts, rows)
    path = paths.results / folder / f"scorecards.{source}.haiku-5-5.jsonl"
    store.write(path, [card.to_json() for card in cards])


def test_render_reports_a_perfect_run(tmp_path, calls, cards):
    write_run(tmp_path, calls, cards)
    text = report.render(tmp_path)
    assert f"{len(calls)} synthetic calls by 8 loan officers" in text
    assert "| Word error rate, numbers excluded | 0.0% (0 errors in" in text
    assert f"| Calls whose speaker order matches the original | 100.0% ({len(calls)}/" in text
    assert "### In detail: claude-haiku-5-5, Whisper transcript" in text
    assert "#### Every call the grader got wrong\n\nNone." in text
    assert "Largest gap in any officer's meaningful-call count: 0." in text
    assert report.render(tmp_path) == text  # the same files always give the same report


def test_render_prefers_the_speech_run_and_lists_both(tmp_path, calls, cards):
    write_run(tmp_path, calls, cards, source="clean")
    write_run(tmp_path, calls, cards, source="asr")
    text = report.render(tmp_path)
    assert text.index("| claude-haiku-5-5, Whisper transcript |") < text.index(
        "| claude-haiku-5-5, original text |"
    )
    assert "### In detail: claude-haiku-5-5, Whisper transcript" in text


def test_render_without_grades_says_so(tmp_path, calls):
    store.write(store.Paths("heldout", tmp_path).calls, [call.to_json() for call in calls])
    assert "No grading runs yet" in report.render(tmp_path)


def test_render_covers_both_sets_and_keeps_the_first_version(tmp_path, calls, cards):
    write_run(tmp_path, calls, cards, name="heldout")
    write_run(tmp_path, calls, cards, name="dev")
    write_run(tmp_path, calls, cards, name="dev", folder="first-version")
    text = report.render(tmp_path)
    assert text.index("## Held-out set") < text.index("## Development set")
    assert "| First version: claude-haiku-5-5, Whisper transcript |" in text
    held_out = text[: text.index("## Development set")]
    assert "First version" not in held_out


def test_masking_table_counts_every_planted_detail(tmp_path, calls, cards):
    write_run(tmp_path, calls, cards)
    planted = sum(len(call.personal) for call in calls)
    lines = report.render(tmp_path).splitlines()
    total = next(line for line in lines if line.startswith("| All |"))
    assert (
        total
        == f"| All | {planted} | 100.0% ({planted}/{planted}) | 100.0% ({planted}/{planted}) |"
    )
    assert masked_segments(calls[0])  # the helper the table is checked against


def test_trap_table_ignores_calls_that_also_contain_the_real_rule_break(calls, cards, rubric):
    """A correct grader raises the flag on such a call, and that is not a failed trap."""
    label = "Officer declines to guarantee approval (must not raise guaranteed_approval)"
    calls = copy.deepcopy(calls)
    broke_it = next(call for call in calls if "guaranteed_approval" in call.flags)
    broke_it.turns.append(Turn("agent", report.TRAP_LINES[label][2]))

    run = report.compare("clean", "claude-haiku-5-5", cards, calls, rubric)
    row = next(line for line in report.detail_section(run, calls, rubric) if label in line)
    assert "100.0%" in row
