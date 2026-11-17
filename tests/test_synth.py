import re

from callqa.synth import COVERED, OFFICERS, SKIPPED, TRAPS, VIOLATIONS, generate


def test_generation_is_reproducible():
    first = [call.to_json() for call in generate(count=40, seed=3)]
    assert first == [call.to_json() for call in generate(count=40, seed=3)]
    assert first != [call.to_json() for call in generate(count=40, seed=4)]


def test_every_officer_gets_calls_and_ids_are_unique(calls):
    assert {call.officer for call in calls} == {officer[0] for officer in OFFICERS}
    assert len({call.call_id for call in calls}) == len(calls)


def test_only_conversations_carry_step_labels(calls, rubric):
    for call in calls:
        if call.outcome == "conversation":
            assert set(call.steps) == {step.id for step in rubric.steps_for(call.campaign)}
        else:
            assert call.steps == {} and call.flags == [] and call.personal == []


def matches(template: str, text: str) -> bool:
    """Whether ``text`` is ``template`` with its {fields} filled in."""
    pattern = ".+".join(re.escape(part) for part in re.split(r"\{\w+\}", template))
    return re.fullmatch(pattern, text) is not None


def test_dialogue_matches_the_labels(calls):
    """A covered step's line is in the call, and a skipped step's covering lines are not."""
    for call in (c for c in calls if c.outcome == "conversation"):
        agent = [turn.text for turn in call.turns if turn.speaker == "agent"]
        for step, done in call.steps.items():
            key = f"purpose_{call.campaign}" if step == "purpose" else step
            said = any(matches(line, text) for text in agent for line, _ in COVERED[key])
            assert said == done, (call.call_id, step)


def test_violations_are_spoken_only_when_labelled(calls):
    for call in calls:
        agent = {turn.text for turn in call.turns if turn.speaker == "agent"}
        for flag, lines in VIOLATIONS.items():
            assert bool(agent & set(lines)) == (flag in call.flags), (call.call_id, flag)


def test_traps_and_personal_details_are_present_in_the_set(calls):
    agent = {turn.text for call in calls for turn in call.turns if turn.speaker == "agent"}
    assert all(trap[1] in agent for trap in TRAPS)
    assert SKIPPED["nmls_id"][1][0] in agent
    assert {item["kind"] for call in calls for item in call.personal} == {
        "phone",
        "ssn",
        "email",
        "dob",
    }


def test_personal_details_are_said_by_the_contact(calls):
    for call in calls:
        contact = [turn.text for turn in call.turns if turn.speaker == "customer"]
        for item in call.personal:
            assert item["spoken"] in contact
