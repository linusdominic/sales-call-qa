"""Mask personal details in a transcript before it leaves the machine.

Contacts volunteer phone numbers, email addresses, dates of birth and sometimes a Social
Security number. None of that is needed to grade a call against a script, so it is replaced
with a placeholder before the transcript is sent to a model. The loan officer's NMLS number
is a public licence identifier and is left alone: the script requires it to be said.
"""

import re

from callqa.audio import Segment

# Digits spoken one at a time come out of speech recognition in many shapes: "555-0142",
# "5 5 5 0 1 4 2", "912, 44, 5537", "9-1-2. 4/4, 5/5". This matches a run of digits with
# light punctuation between them.
_DIGIT_RUN = re.compile(r"(?<![\w$.])\d(?:[\s,.\-/]{0,2}\d){6,}(?![\w%])")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SPOKEN_EMAIL = re.compile(
    r"\b[\w.]+(?: dot [\w.]+)* at [\w-]+ dot (?:com|org|net)\b", re.IGNORECASE
)
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_YEAR = r"(?:19|20)\d{2}"
_DATE = re.compile(
    rf"\b(?:(?:{_MONTHS})\.? {_DAY}[.,]? {_YEAR}|(?:the )?{_DAY} of (?:{_MONTHS}),? {_YEAR})\b"
)
# When the words before it say what it is, a bare year is enough to mask. This covers a
# month and day that speech recognition has garbled beyond what a date pattern can match.
_BIRTH_YEAR = re.compile(
    rf"(\b(?:date of birth|birth ?date|birthday|born)\b.{{0,40}}?)\b{_YEAR}\b", re.I
)


def _digit_label(match: re.Match[str]) -> str:
    digits = sum(character.isdigit() for character in match.group())
    if digits == 9:
        return "[SSN]"
    if digits in (7, 10, 11):
        return "[PHONE]"
    return "[NUMBER]"


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Return the masked text and how many of each kind were masked."""
    counts: dict[str, int] = {}

    def swap(pattern: re.Pattern[str], label) -> None:
        nonlocal text

        def replace(match: re.Match[str]) -> str:
            tag = label(match) if callable(label) else label
            counts[tag] = counts.get(tag, 0) + 1
            return tag

        text = pattern.sub(replace, text)

    swap(_EMAIL, "[EMAIL]")
    swap(_SPOKEN_EMAIL, "[EMAIL]")
    swap(_DATE, "[DATE]")
    text, years = _BIRTH_YEAR.subn(r"\1[DATE]", text)
    if years:
        counts["[DATE]"] = counts.get("[DATE]", 0) + years
    swap(_DIGIT_RUN, _digit_label)
    return text, counts


def mask(segments: list[Segment]) -> list[Segment]:
    """A transcript with the personal details in every line masked."""
    return [Segment(s.speaker, s.start, s.end, redact(s.text)[0]) for s in segments]


def leaked(kind: str, value: str, text: str) -> bool:
    """Whether a planted personal detail is still readable in ``text``."""
    if kind in ("phone", "ssn"):
        return value in re.sub(r"\D", "", text)
    if kind == "email":
        name = value.split("@")[0].replace(".", " ")
        flat = re.sub(r"[^a-z0-9@]+", " ", text.lower())
        return "@example" in flat or (name in flat and "example" in flat)
    year = value[-4:]  # date of birth: the year is the identifying part
    return year in text
