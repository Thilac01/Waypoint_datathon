# Final package verification

The final package was checked after the decision-evidence and presentation updates.

| Check | Result |
|---|---|
| Original prediction/allocation CSVs | All 3 byte-for-byte unchanged |
| Original trained models | All 4 byte-for-byte unchanged |
| Official allocation checker source | Byte-for-byte identical to supplied file |
| Fresh-first, fairness-first, balanced | All OPTIMAL, 0% gap, original checker PASSED |
| Separate maximum-order-count solve | 79 orders; objective equals bound |
| Notebook execution | 16 code cells executed in order, no errors |
| Final saved-model inference cell | Both prediction files reproduced |
| Existing verification tests | 5 / 5 passed |
| Decision showcase | One-page PDF; visually inspected |
| Demo review | 4:10.18, 1920 × 1080, H.264 video, AAC audio, English captions |
| Audio signal | Present; peak −2.7 dBFS; no digital clipping indicated |
| Dataset packaging | Original organizer CSV datasets excluded |

Evidence is in `outputs/reports/final_integrity.json`, `test_run.log`, `notebook_execution.log`, `submission_freeze.json`, `policy_sensitivity.json`, the per-policy checker logs, and `DEMO_CHAPTERS.json`.

The video is a narrated replay of real executed notebook/report excerpts with synthetic speech. It is a review draft, not a live human recording. Human review, eligibility determination, the unlisted YouTube upload and competition submission remain the team's responsibility. No upload, publication or submission has been performed.
