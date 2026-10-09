# Suggested 3–5 minute demo

1. **0:00–0:35 — problem and outputs.** Show the three generated submission files. Explain service time, arrival-lateness probability, ten-week depot demand and the peak-day plan.
2. **0:35–1:15 — labels.** Show the notebook's route/order join and the 04:48 arrival, 05:00 opening, 05:12 departure example. Explain why handling is 12 minutes. Point out that arrival exactly at close is not late.
3. **1:15–2:10 — modelling and validation.** Show the architecture and actual leaderboards. Explain chronological tuning and untouched holdouts, explicit model comparisons, probability calibration and why random splits would be inappropriate. Show per-brand/depot errors as well as pooled results.
4. **2:10–2:50 — demand.** Show that deferred and never-dispatched orders count on the requested date. Explain ten-week forecast-origin backtests and the rule that only Fresh has chilled demand.
5. **2:50–3:40 — allocation.** Show the official checker pass, 79 served orders, the refrigerated fleet bottleneck, trip-time audit and six deferrals. Explain that the policy objective is a team choice and distinguish individual impossibility from shared-resource tradeoffs.
6. **3:40–4:20 — inference and limitations.** Run the final notebook cell loading saved models. State actual holdout metrics, that hidden-test performance is unknown, and the role of AI assistance.

Record the final demo yourself and upload it as an unlisted YouTube video. Replace this outline with the final link in your submission materials.
