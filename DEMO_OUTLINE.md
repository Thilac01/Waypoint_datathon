# Alt-F4 demo storyline

The MP4 is a review draft: synthetic narration over a replay of real executed notebook outputs, not a live screen recording. The team must understand and review it. For a human recording, open `Alt-F4_FinalNotebook.ipynb` or its HTML export and follow this script. Show the specified real cells and reports; do not substitute fabricated results.

## 00:00:00 — A dispatch decision we can defend

This is Alt F four's Explainable Logistics Decision Intelligence submission. The business question is not just which model has the best score. It is which orders can be delivered, with the fleet actually available. Our verified solution serves seventy nine of eighty five orders. This walkthrough uses real executed notebook outputs. The narration is synthetic and requires team review.

## 00:00:21 — Correct labels come before clever models

First, the notebook constructs waiting aware service labels. If a vehicle arrives at eight forty, the outlet opens at nine, and handling ends at nine twelve, service is twelve minutes, not thirty two. The twenty minute wait is excluded. Arrival after window close is late; arrival exactly at close is not. We join dispatched orders to their route legs one to one, exclude orders that never ran, and remove actual outcomes before building planned features.

## 00:00:47 — Train in the past; assess in the future

We compare eight explicit configurations for each delivery target, including boosted trees, extra trees, and linear baselines. Two earlier four week folds choose models and ensemble weights. The six week final holdout comes later, from January fourth to February fourteenth. Service prediction blends CatBoost and histogram boosting. Late probability blends X G Boost, CatBoost and Light G B M. The displayed results are from that frozen holdout, not the training data.

## 00:01:14 — Make performance differences visible

Service mean absolute error is three point seven five minutes, and lateness area under the curve is zero point nine seven five. But the brand and depot table matters. Fresh dominates the sample. Tech has larger absolute handling errors, and its segment samples are much smaller. For operational review, one thousand and seventeen of the five thousand and fourteen test orders have predicted late probability of at least fifty percent. These are risks, not observed delays.

## 00:01:39 — Forecast demand before committing capacity

Demand uses order date and includes every order, even deferred orders and orders that never ran. We build complete weekly series for each depot and brand, and forecast ten weeks from a fixed origin. Candidate models include rolling means, a seasonal baseline, ridge regression and boosting. The submitted network forecast peaks in week fifteen at about two thousand and seven cubic metres, including six hundred and eighty two chilled. This supports inventory and refrigerated availability planning.

## 00:02:07 — Show the weakness; avoid last-minute overfitting

Overall total demand weighted absolute percentage error is six point zero eight percent. That hides Tech errors of twenty four point seven percent in Kandy and twenty nine point eight percent in Peliyagoda. Several series have negative R squared. We tested nine simple Tech baselines using only earlier chronological folds. The Kandy bias correction improved validation error by less than one percent; the Peliyagoda alternatives were worse. We kept the verified predictions and report these limitations openly.

## 00:02:35 — A physical bottleneck, then a feasibility proof

The peak day needs one hundred and eighty one point six cubic metres of chilled capacity. Four available refrigerated vehicles can carry at most one hundred and seventy two point four cubic metres across two full volume trips each. That is an optimistic bound: demand already exceeds it by nine point two cubic metres. C P Sat then enforces whole orders, depot, brand and district compatibility, van access, weight, volume and published time budgets. A separate maximum count solve proves the ceiling of seventy nine deliveries.

## 00:03:05 — Policy changes have a measurable cost

We changed only the priority weights, leaving every hard constraint unchanged. Fresh first and balanced both serve seventy nine orders and deliver one hundred and forty point seven cubic metres chilled. Fairness first serves seventy seven, but clears all ten previously deferred orders instead of nine. That choice sacrifices two deliveries and fourteen point four six nine cubic metres of chilled volume. The comparison uses common reference points, because different policy scores have different scales. All three solutions are optimal and pass the original checker.

## 00:03:36 — Reproduce the result and state the limits

The final notebook cell reloads the saved models and reproduces both prediction files. Template checks preserve every identifier, column and row order. The allocation checker passes, and all three original submission files remain unchanged. These task horizons are distinct; they are not one jointly solved real world route plan. Optimality covers the published competition constraints, not unmodelled traffic, detailed arrival windows or remaining weekly fuel. The package discloses the A I assistance and requires human team review. Our contribution is a decision that can be explained, checked and defended.

## Finish before submission

Review the disclosure and limitations; check the final inference cell and official checker result. Upload the reviewed 3–5 minute video to YouTube as unlisted and provide its actual link in the competition form. No video has been uploaded or submission made by this code. Do not publish organizer datasets.
