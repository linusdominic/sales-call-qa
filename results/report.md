# Evaluation report

Built by `callqa evaluate` from the files in `data/` and `results/`. No model is called.

## The calls

160 synthetic calls by 8 loan officers: 14 callback_requested, 80 conversation, 12 declined_early, 42 voicemail, 12 wrong_number. By the rule in `config/script.yml`, 68 are meaningful. 12 contain at least one rule break.

## Transcription

160 two-channel recordings, 120 minutes of audio, transcribed with whisper.cpp `base.en` one channel at a time.

| Measure | Result |
|---|---|
| Word error rate, numbers excluded | 7.0% (1057 errors in 15189 words) |
| Calls whose speaker order matches the original | 100.0% (160/160) |

## Masking of personal details

Details the contacts say aloud, and how many were unreadable after masking.

| Detail | Planted | Masked in Whisper transcript | Masked in original text |
|---|---|---|---|
| Phone number | 14 | 100.0% (14/14) | 100.0% (14/14) |
| Social Security number | 17 | 94.1% (16/17) | 100.0% (17/17) |
| Email address | 14 | 100.0% (14/14) | 100.0% (14/14) |
| Date of birth | 17 | 52.9% (9/17) | 100.0% (17/17) |
| All | 62 | 85.5% (53/62) | 100.0% (62/62) |

## Grading against the true labels

| Run | Outcome | Meaningful: precision | Meaningful: recall | Script steps | Flags: precision | Flags: recall | Calls with no error | Cost | Median time |
|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5, Whisper transcript | 100.0% (158/158) | 98.5% (66/67) | 100.0% (66/66) | 99.7% (778/780) | 60.0% (15/25) | 100.0% (15/15) | 92.4% (146/158) | $0.08 | 2.4 s |
| claude-haiku-5-5, original text | 100.0% (160/160) | 98.6% (68/69) | 100.0% (68/68) | 99.4% (795/800) | 78.9% (15/19) | 100.0% (15/15) | 94.4% (151/160) | $0.08 | 2.4 s |

Script steps are counted over the calls that really were conversations: one decision per step per call. Flags are counted over every call.

## In detail: claude-haiku-5-5, Whisper transcript

- Outcome: 100.0% (158/158), 95% CI 98 to 100%.
- Meaningful or not: 99.4% (157/158), 95% CI 97 to 100%. 66 true positives, 1 false positives, 0 false negatives, 91 true negatives.
- Script step decisions: 99.7% (778/780), 95% CI 99 to 100%.
- Quoted evidence found in the transcript: 100.0% (705/705).
- Cost: $0.0772 for 158 calls, $0.49 per 1,000 calls. Time per call: median 2.4 s, slowest 4.6 s.

### By script step

| Step | Decisions | Correct | Graded covered, was missing | Graded missing, was covered |
|---|---|---|---|---|
| Identifies self and company | 78 | 100.0% (78/78) | 0 | 0 |
| Recording disclosure | 78 | 100.0% (78/78) | 0 | 0 |
| States NMLS ID | 78 | 100.0% (78/78) | 0 | 0 |
| States the purpose of the call | 78 | 100.0% (78/78) | 0 | 0 |
| Asks if it is a good time | 40 | 100.0% (40/40) | 0 | 0 |
| Confirms the inquiry details | 38 | 100.0% (38/38) | 0 | 0 |
| Asks about the loan goal | 78 | 98.7% (77/78) | 1 | 0 |
| Asks about the property or current loan | 78 | 100.0% (78/78) | 0 | 0 |
| Asks about credit or income | 78 | 100.0% (78/78) | 0 | 0 |
| Secures a next step | 78 | 100.0% (78/78) | 0 | 0 |
| Closes with a callback number | 78 | 98.7% (77/78) | 0 | 1 |

### Trap lines

| Trap | Calls | Handled correctly |
|---|---|---|
| Officer declines to guarantee approval (must not raise guaranteed_approval) | 10 | 80.0% (8/10) |
| Officer declines to promise a rate (must not raise rate_promise) | 11 | 81.8% (9/11) |
| Officer says "no rush" (must not raise high_pressure) | 10 | 70.0% (7/10) |
| Officer says "licensed" with no number (nmls_id must be missing) | 6 | 100.0% (6/6) |
| Contact asks about recording, officer dodges (recording_disclosure must be missing) | 5 | 100.0% (5/5) |
| Officer says "reach out if you want" (next_step must be missing) | 11 | 100.0% (11/11) |

### Review queue

A call goes to a person when it has a compliance flag, the model was not confident, or a quote could not be found in the transcript.

- Calls sent for review: 34.2% (54/158).
- Of those, calls with a real rule break or a grading error: 37.0% (20/54).
- Calls with a grading error: 12. Caught by the queue: 91.7% (11/12).
- Calls with a real rule break: 12. In the queue: 100.0% (12/12).

### What the dashboard shows per loan officer, against the truth

| Loan officer | Calls | Meaningful: true | graded | Script adherence: true | graded | Flags: true | graded |
|---|---|---|---|---|---|---|---|
| Daniel Okafor | 20 | 13 | 13 | 84% | 83% | 0 | 1 |
| Grace Kim | 19 | 7 | 7 | 90% | 90% | 5 | 6 |
| James Carter | 20 | 9 | 9 | 91% | 91% | 0 | 0 |
| Marcus Hill | 20 | 6 | 6 | 76% | 76% | 0 | 2 |
| Maria Lopez | 20 | 9 | 9 | 98% | 98% | 0 | 0 |
| Priya Nair | 19 | 8 | 8 | 88% | 88% | 0 | 2 |
| Sofia Reyes | 20 | 7 | 8 | 82% | 83% | 0 | 2 |
| Tom Becker | 20 | 7 | 7 | 92% | 92% | 10 | 12 |

Largest gap in any officer's meaningful-call count: 1.

### Every call the grader got wrong

- `call_003` (Priya Nair, warm): false flag rate_promise. In review queue.
- `call_006` (Tom Becker, warm): false flag guaranteed_approval. In review queue.
- `call_008` (Marcus Hill, warm): false flag rate_promise. In review queue.
- `call_060` (Daniel Okafor, cold): closing graded missing. Not in review queue.
- `call_061` (Sofia Reyes, cold): false flag guaranteed_approval. In review queue.
- `call_069` (Sofia Reyes, cold): graded meaningful, was not; qualify_goal graded covered. In review queue.
- `call_071` (Grace Kim, warm): false flag high_pressure. In review queue.
- `call_076` (Daniel Okafor, cold): false flag high_pressure. In review queue.
- `call_086` (Tom Becker, warm): false flag guaranteed_approval. In review queue.
- `call_120` (Marcus Hill, cold): false flag high_pressure. In review queue.
- `call_131` (Priya Nair, warm): false flag high_pressure. In review queue.
- `call_141` (Sofia Reyes, cold): false flag high_pressure. In review queue.
