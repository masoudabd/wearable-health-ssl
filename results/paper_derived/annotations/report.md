# Recordings, annotated episodes and window annotations

Written by `scripts/paper_annotations.py` on 2026-10-04 at commit f14a29c (post hoc, read-only).
Every kept window was checked against `windows_30hz_5s_hop2.5.npz` (start time, label and FOG fraction):
59,250 windows, all identical.

| Cohort | Recordings | Hours recorded | Hours with valid labels | Candidate windows | Kept windows | Excluded windows | Annotated episodes | Episodes entirely in valid time |
|---|---|---|---|---|---|---|---|---|
| defog | 183 | 86.5 | 27.0 | 124,279 | 31,894 | 92,385 | 3,097 | 2,036 |
| tdcsfog | 1,039 | 20.1 | 20.1 | 27,356 | 27,356 | 0 | 1,794 | 1,794 |

A window is excluded when any of its samples is invalid (home cohort: outside a protocol task or marked ambiguous
by the dataset's Valid flag; lab test recordings: samples the official solution marks Ignored). An episode is a run
of consecutive FOG samples of one class at the native sampling rate; the notype recordings have no class.
