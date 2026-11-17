# Evaluation report

Built by `callqa evaluate` from the files in `data/` and `results/`. No model is called.

The held-out set was generated after the grader's prompt and the masking rules were frozen. The development set is the one they were worked out on, so its figures for the current version are optimistic.

## Held-out set

160 synthetic calls by 8 loan officers: 12 callback_requested, 90 conversation, 11 declined_early, 34 voicemail, 13 wrong_number. By the rule in `config/script.yml`, 71 are meaningful. 13 contain at least one rule break.

### Transcription

160 two-channel recordings, 131 minutes of audio, transcribed with whisper.cpp `base.en` one channel at a time.

| Measure | Result |
|---|---|
| Word error rate, numbers excluded | 6.4% (1054 errors in 16422 words) |
| Calls whose speaker order matches the original | 100.0% (160/160) |

### Masking of personal details

Details the contacts say aloud, and how many were unreadable after masking.

| Detail | Planted | Masked in Whisper transcript | Masked in original text |
|---|---|---|---|
| Phone number | 21 | 100.0% (21/21) | 100.0% (21/21) |
| Social Security number | 19 | 100.0% (19/19) | 100.0% (19/19) |
| Email address | 15 | 100.0% (15/15) | 100.0% (15/15) |
| Date of birth | 19 | 100.0% (19/19) | 100.0% (19/19) |
| All | 74 | 100.0% (74/74) | 100.0% (74/74) |

### Grading against the true labels

| Run | Outcome | Meaningful: precision | Meaningful: recall | Script steps | Flags: precision | Flags: recall | Calls with no error | Sent for review | Cost |
|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5, Whisper transcript | 100.0% (160/160) | 100.0% (71/71) | 100.0% (71/71) | 99.8% (898/900) | 77.8% (14/18) | 100.0% (14/14) | 96.2% (154/160) | 17.5% (28/160) | $0.10 |
| claude-sonnet-5-5, Whisper transcript | 100.0% (160/160) | 100.0% (71/71) | 100.0% (71/71) | 100.0% (900/900) | 77.8% (14/18) | 100.0% (14/14) | 97.5% (156/160) | 25.6% (41/160) | $1.72 |
| claude-haiku-5-5, original text | 100.0% (160/160) | 100.0% (71/71) | 100.0% (71/71) | 100.0% (900/900) | 100.0% (14/14) | 100.0% (14/14) | 100.0% (160/160) | 10.0% (16/160) | $0.09 |

Script steps are counted over the calls that really were conversations: one decision per step per call. Flags are counted over every call.

### In detail: claude-haiku-5-5, Whisper transcript

- Outcome: 100.0% (160/160), 95% CI 98 to 100%.
- Meaningful or not: 100.0% (160/160), 95% CI 98 to 100%. 71 true positives, 0 false positives, 0 false negatives, 89 true negatives.
- Script step decisions: 99.8% (898/900), 95% CI 99 to 100%.
- Quoted evidence found in the transcript: 100.0% (805/805).
- Cost: $0.0995 for 160 calls, $0.62 per 1,000 calls. Time per call: median 2.8 s, slowest 5.8 s.

#### By script step

| Step | Decisions | Correct | Graded covered, was missing | Graded missing, was covered |
|---|---|---|---|---|
| Identifies self and company | 90 | 98.9% (89/90) | 0 | 1 |
| Recording disclosure | 90 | 100.0% (90/90) | 0 | 0 |
| States NMLS ID | 90 | 100.0% (90/90) | 0 | 0 |
| States the purpose of the call | 90 | 100.0% (90/90) | 0 | 0 |
| Asks if it is a good time | 34 | 100.0% (34/34) | 0 | 0 |
| Confirms the inquiry details | 56 | 100.0% (56/56) | 0 | 0 |
| Asks about the loan goal | 90 | 100.0% (90/90) | 0 | 0 |
| Asks about the property or current loan | 90 | 100.0% (90/90) | 0 | 0 |
| Asks about credit or income | 90 | 100.0% (90/90) | 0 | 0 |
| Secures a next step | 90 | 100.0% (90/90) | 0 | 0 |
| Closes with a callback number | 90 | 98.9% (89/90) | 0 | 1 |

#### Trap lines

| Trap | Calls | Handled correctly |
|---|---|---|
| Officer declines to guarantee approval (must not raise guaranteed_approval) | 18 | 77.8% (14/18) |
| Officer declines to promise a rate (must not raise rate_promise) | 10 | 100.0% (10/10) |
| Officer says "no rush" (must not raise high_pressure) | 15 | 100.0% (15/15) |
| Officer says "licensed" with no number (nmls_id must be missing) | 7 | 100.0% (7/7) |
| Contact asks about recording, officer dodges (recording_disclosure must be missing) | 6 | 100.0% (6/6) |
| Officer says "reach out if you want" (next_step must be missing) | 13 | 100.0% (13/13) |

#### Review queue

A call goes to a person when it has a compliance flag, the model was not confident, or a quote could not be found in the transcript.

- Calls sent for review: 17.5% (28/160).
- Of those, calls with a real rule break or a grading error: 64.3% (18/28).
- Calls with a grading error: 6. Caught by the queue: 100.0% (6/6).
- Calls with a real rule break: 13. In the queue: 100.0% (13/13).

#### What the dashboard shows per loan officer, against the truth

| Loan officer | Calls | Meaningful: true | graded | Script adherence: true | graded | Flags: true | graded |
|---|---|---|---|---|---|---|---|
| Daniel Okafor | 20 | 12 | 12 | 92% | 92% | 0 | 0 |
| Grace Kim | 20 | 9 | 9 | 92% | 92% | 4 | 4 |
| James Carter | 20 | 11 | 11 | 87% | 86% | 0 | 0 |
| Marcus Hill | 20 | 9 | 9 | 83% | 83% | 0 | 0 |
| Maria Lopez | 20 | 8 | 8 | 98% | 98% | 0 | 0 |
| Priya Nair | 20 | 7 | 7 | 91% | 91% | 0 | 1 |
| Sofia Reyes | 20 | 6 | 6 | 75% | 75% | 0 | 3 |
| Tom Becker | 20 | 9 | 9 | 88% | 88% | 10 | 10 |

Largest gap in any officer's meaningful-call count: 0.

#### Every call the grader got wrong

- `call_013` (Sofia Reyes, warm): false flag guaranteed_approval. In review queue.
- `call_014` (Tom Becker, warm): closing graded missing. In review queue.
- `call_029` (Sofia Reyes, warm): false flag guaranteed_approval. In review queue.
- `call_053` (Sofia Reyes, warm): false flag guaranteed_approval. In review queue.
- `call_058` (James Carter, cold): greeting_identify graded missing. In review queue.
- `call_099` (Priya Nair, cold): false flag guaranteed_approval. In review queue.

## Development set

160 synthetic calls by 8 loan officers: 14 callback_requested, 80 conversation, 12 declined_early, 42 voicemail, 12 wrong_number. By the rule in `config/script.yml`, 68 are meaningful. 12 contain at least one rule break.

### Transcription

160 two-channel recordings, 120 minutes of audio, transcribed with whisper.cpp `base.en` one channel at a time.

| Measure | Result |
|---|---|
| Word error rate, numbers excluded | 7.0% (1057 errors in 15189 words) |
| Calls whose speaker order matches the original | 100.0% (160/160) |

### Masking of personal details

Details the contacts say aloud, and how many were unreadable after masking.

| Detail | Planted | Masked in Whisper transcript | Masked in original text |
|---|---|---|---|
| Phone number | 14 | 100.0% (14/14) | 100.0% (14/14) |
| Social Security number | 17 | 100.0% (17/17) | 100.0% (17/17) |
| Email address | 14 | 100.0% (14/14) | 100.0% (14/14) |
| Date of birth | 17 | 100.0% (17/17) | 100.0% (17/17) |
| All | 62 | 100.0% (62/62) | 100.0% (62/62) |

### Grading against the true labels

| Run | Outcome | Meaningful: precision | Meaningful: recall | Script steps | Flags: precision | Flags: recall | Calls with no error | Sent for review | Cost |
|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5, Whisper transcript | 100.0% (160/160) | 100.0% (68/68) | 100.0% (68/68) | 100.0% (800/800) | 93.8% (15/16) | 100.0% (15/15) | 99.4% (159/160) | 16.2% (26/160) | $0.10 |
| claude-haiku-5-5, original text | 100.0% (160/160) | 100.0% (68/68) | 100.0% (68/68) | 100.0% (800/800) | 100.0% (15/15) | 100.0% (15/15) | 100.0% (160/160) | 8.8% (14/160) | $0.09 |
| First version: claude-haiku-5-5, Whisper transcript | 100.0% (158/158) | 98.5% (66/67) | 100.0% (66/66) | 99.7% (778/780) | 60.0% (15/25) | 100.0% (15/15) | 92.4% (146/158) | 34.2% (54/158) | $0.08 |
| First version: claude-haiku-5-5, original text | 100.0% (160/160) | 98.6% (68/69) | 100.0% (68/68) | 99.4% (795/800) | 78.9% (15/19) | 100.0% (15/15) | 94.4% (151/160) | 9.4% (15/160) | $0.08 |

Script steps are counted over the calls that really were conversations: one decision per step per call. Flags are counted over every call.

### In detail: claude-haiku-5-5, Whisper transcript

- Outcome: 100.0% (160/160), 95% CI 98 to 100%.
- Meaningful or not: 100.0% (160/160), 95% CI 98 to 100%. 68 true positives, 0 false positives, 0 false negatives, 92 true negatives.
- Script step decisions: 100.0% (800/800), 95% CI 100 to 100%.
- Quoted evidence found in the transcript: 100.0% (716/716).
- Cost: $0.0986 for 160 calls, $0.62 per 1,000 calls. Time per call: median 2.8 s, slowest 6.7 s.

#### By script step

| Step | Decisions | Correct | Graded covered, was missing | Graded missing, was covered |
|---|---|---|---|---|
| Identifies self and company | 80 | 100.0% (80/80) | 0 | 0 |
| Recording disclosure | 80 | 100.0% (80/80) | 0 | 0 |
| States NMLS ID | 80 | 100.0% (80/80) | 0 | 0 |
| States the purpose of the call | 80 | 100.0% (80/80) | 0 | 0 |
| Asks if it is a good time | 40 | 100.0% (40/40) | 0 | 0 |
| Confirms the inquiry details | 40 | 100.0% (40/40) | 0 | 0 |
| Asks about the loan goal | 80 | 100.0% (80/80) | 0 | 0 |
| Asks about the property or current loan | 80 | 100.0% (80/80) | 0 | 0 |
| Asks about credit or income | 80 | 100.0% (80/80) | 0 | 0 |
| Secures a next step | 80 | 100.0% (80/80) | 0 | 0 |
| Closes with a callback number | 80 | 100.0% (80/80) | 0 | 0 |

#### Trap lines

| Trap | Calls | Handled correctly |
|---|---|---|
| Officer declines to guarantee approval (must not raise guaranteed_approval) | 9 | 88.9% (8/9) |
| Officer declines to promise a rate (must not raise rate_promise) | 11 | 100.0% (11/11) |
| Officer says "no rush" (must not raise high_pressure) | 10 | 100.0% (10/10) |
| Officer says "licensed" with no number (nmls_id must be missing) | 6 | 100.0% (6/6) |
| Contact asks about recording, officer dodges (recording_disclosure must be missing) | 5 | 100.0% (5/5) |
| Officer says "reach out if you want" (next_step must be missing) | 11 | 100.0% (11/11) |

#### Review queue

A call goes to a person when it has a compliance flag, the model was not confident, or a quote could not be found in the transcript.

- Calls sent for review: 16.2% (26/160).
- Of those, calls with a real rule break or a grading error: 50.0% (13/26).
- Calls with a grading error: 1. Caught by the queue: 100.0% (1/1).
- Calls with a real rule break: 12. In the queue: 100.0% (12/12).

#### What the dashboard shows per loan officer, against the truth

| Loan officer | Calls | Meaningful: true | graded | Script adherence: true | graded | Flags: true | graded |
|---|---|---|---|---|---|---|---|
| Daniel Okafor | 20 | 13 | 13 | 84% | 84% | 0 | 0 |
| Grace Kim | 20 | 8 | 8 | 91% | 91% | 5 | 5 |
| James Carter | 20 | 9 | 9 | 91% | 91% | 0 | 0 |
| Marcus Hill | 20 | 6 | 6 | 76% | 76% | 0 | 0 |
| Maria Lopez | 20 | 9 | 9 | 98% | 98% | 0 | 0 |
| Priya Nair | 20 | 9 | 9 | 89% | 89% | 0 | 0 |
| Sofia Reyes | 20 | 7 | 7 | 82% | 82% | 0 | 1 |
| Tom Becker | 20 | 7 | 7 | 92% | 92% | 10 | 10 |

Largest gap in any officer's meaningful-call count: 0.

#### Every call the grader got wrong

- `call_061` (Sofia Reyes, cold): false flag guaranteed_approval. In review queue.
