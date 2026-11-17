import pytest
from conftest import masked_segments

from callqa.redact import leaked, redact


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The best number for me is 555-0142.", "The best number for me is [PHONE]."),
        ("The best number for me is 5 5 5, 0 1 4 2.", "The best number for me is [PHONE]."),
        ("Call 555 010 4477 tomorrow", "Call [PHONE] tomorrow"),
        ("It's 912-44-5537.", "It's [SSN]."),
        ("It's 9 1 2, 4 4, 5 5 3 7.", "It's [SSN]."),
        ("Email jordan.rivera@example.com please", "Email [EMAIL] please"),
        ("jordan dot rivera at example dot com", "[EMAIL]"),
        ("My date of birth is March 14, 1966.", "My date of birth is [DATE]."),
    ],
)
def test_masks_personal_details(text, expected):
    assert redact(text)[0] == expected


@pytest.mark.parametrize(
    "text",
    [
        "My NMLS number is 603288.",  # a public licence number, and the script requires it
        "It's worth about $400,000 and I owe around 250,000.",
        "My rate is 7.1% and I owe about 300,000.",
        "Does Thursday at 10 work?",
        "Last I checked it was around 720.",
    ],
)
def test_leaves_business_numbers_alone(text):
    assert redact(text) == (text, {})


def test_counts_what_was_masked():
    _, counts = redact("Reach me at 555-0142 or 555-0143, or jo@example.com.")
    assert counts == {"[PHONE]": 2, "[EMAIL]": 1}


def test_no_planted_detail_survives_masking_of_the_original_text(calls):
    planted = 0
    for call in calls:
        text = " ".join(segment.text for segment in masked_segments(call))
        for item in call.personal:
            planted += 1
            assert not leaked(item["kind"], item["value"], text), (call.call_id, item)
    assert planted > 10


def test_leak_check_sees_an_unmasked_detail(calls):
    call = next(call for call in calls if call.personal)
    item = call.personal[0]
    spoken_as_text = item["value"]
    assert leaked(item["kind"], item["value"], f"they said {spoken_as_text} on the call")
