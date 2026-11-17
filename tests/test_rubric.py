from callqa.rubric import Rubric


def test_script_loads_with_campaign_specific_steps(rubric: Rubric):
    cold = [step.id for step in rubric.steps_for("cold")]
    warm = [step.id for step in rubric.steps_for("warm")]
    assert "permission" in cold and "permission" not in warm
    assert "confirm_inquiry" in warm and "confirm_inquiry" not in cold
    assert len(cold) == len(warm) == 10


def test_meaningful_needs_a_conversation(rubric: Rubric):
    covered = {"purpose", "qualify_goal", "qualify_property", "qualify_credit"}
    assert rubric.is_meaningful("conversation", covered)
    assert not rubric.is_meaningful("voicemail", covered)
    assert not rubric.is_meaningful("callback_requested", covered)


def test_meaningful_needs_purpose_and_two_qualifying_questions(rubric: Rubric):
    assert rubric.is_meaningful("conversation", {"purpose", "qualify_goal", "qualify_credit"})
    assert not rubric.is_meaningful("conversation", {"purpose", "qualify_goal"})
    assert not rubric.is_meaningful("conversation", {"qualify_goal", "qualify_credit"})
    # Steps outside the rule do not make up for a missing qualifying question.
    polite = {"purpose", "greeting_identify", "closing", "next_step", "qualify_goal"}
    assert not rubric.is_meaningful("conversation", polite)


def test_adherence_is_share_of_the_campaigns_steps(rubric: Rubric):
    assert rubric.adherence("cold", set()) == 0
    assert rubric.adherence("cold", {s.id for s in rubric.steps_for("cold")}) == 1
    assert rubric.adherence("warm", {"purpose", "closing"}) == 0.2
    # A step that belongs to the other campaign earns nothing.
    assert rubric.adherence("warm", {"permission"}) == 0
