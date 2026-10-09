# Data preparation and label construction

## Sources and row semantics

The organizer's 17 CSV files are read without modification. A file manifest records byte counts, row counts and SHA-256 hashes. Unique order IDs, unique route/sequence keys, reference-table cardinalities and full join coverage are enforced. Any invalid mapping raises an error rather than silently dropping rows. Original identifiers and template order are preserved in all predictions.

An order is demand even when it is deferred or never dispatched. A route leg is an observed journey to one delivered order. These are different populations: Task1 uses 91,894 dispatched orders, while Task2A includes all 97,321 orders available across training and Task1 test inputs. The 413 `not_run` orders have no Task1 labels but remain in demand totals.

## Task1 labels

Clock times are in Asia/Colombo. Convert HH:MM into minutes since midnight. The supplied records use same-day windows/completion; the implementation rejects an overnight/inconsistent case rather than assuming its date.

1. Match `route_id` plus `seq_in_route` to route `route_id` plus `seq` and verify the destination outlet, date, depot, vehicle and planned arrival.
2. Receiving starts at the later of actual arrival and window opening. Therefore `service_min = leave_outlet_time - max(arrival_time, window_open_time)`.
3. `is_late = int(arrival_time > window_close_time)`. Arrival exactly at close is not late. Actual completion after close does not by itself define lateness.
4. Exclude never-dispatched orders from Task1; do not invent their labels.

Example from the supplied records: arrival 04:48, window opens 05:00, departure 05:12. Handling is 12 minutes; the earlier 12 minutes are waiting. Focused tests verify this case and the exact-close boundary.

## Prediction-time features

All actual departure, actual travel, actual arrival and actual completion columns are deleted before constructing features. Raw delivery, route and leg identifiers are not model inputs. Outlet ID is retained as a legitimate static entity feature; it is not a target statistic. There is no target encoding across folds.

Features include order units/weight/volume, density and capacity fractions; outlet access and dock; vehicle type and temperature; planned departure/arrival, window slack and planned wait; calendar, payday, festival ramp and season; district travel standards, typical traffic and supplied date-specific road disruptions; route position, planned elapsed time, route totals and cumulative preceding load/travel/allowance features. All route aggregates use known plans, never actual target outcomes. Date-specific disruption is treated as an available planning input because it is supplied for the prediction horizon; a real deployment would require its availability at dispatch time.

Missing festival names become `none`. Reference joins must match. Numeric imputation/encoding for scikit-learn, LightGBM and XGBoost is fitted on each training fold only. CatBoost uses native categorical features. Linear models use one-hot categories and scaled numeric features; other non-CatBoost trees use a training-fitted ordinal encoder with a reserved unknown value. Large legitimate service durations are retained rather than clipped out of the training labels.

## Temporal evaluation and model selection

Task1: two successive four-week tuning periods precede a six-week final holdout. Each fold trains only on older dispatch dates. All records from a route remain together. Early stopping, model ranking, ensemble weights and optional sigmoid calibration use tuning data only. Only the selected frozen ensemble is evaluated on the final holdout. After assessment it is refitted using all labeled history for submission.

Models are selected by service RMSE and lateness log loss because the booklet does not state exact organizer metrics. MAE, R², WAPE, sMAPE, Brier score, ROC-AUC, PR-AUC and threshold accuracy are also recorded where applicable. These metrics are not interchangeable; regression error is not an accuracy percentage.

Task2A: create complete daily depot/brand panels by `order_date`, including every status. Use calendar-provided ISO year/week, not Gregorian year paired with ISO week. Days with no orders have zero demand. Reject partial observed weeks so a short week cannot masquerade as low demand. Weekly totals sum exactly to all historical order volumes.

Demand features use known calendar fields, weekday interactions and annual Fourier terms. Daily forecasts sum to weeks; weekly models predict volumes directly. Last-4/13/26-week means and a 52-week seasonal baseline provide checks against unnecessary complexity. Two frozen ten-week backtests select a blend separately for each depot/brand/target. The final ten weeks are untouched during selection. Mean/seasonal baseline references remain fixed at the fold's forecast origin; no actuals from within the horizon enter predictions. Total and chilled predictions are nonnegative; non-Fresh chilled is exactly zero; chilled cannot exceed total.

## Limits and reproducibility

All models are fitted from scratch on supplied synthetic data. The candidate lists and hyperparameters are explicit; this is not a pretrained or external API solution. Seeds, configuration, package versions, input hashes, selected weights, boosting rounds and holdout predictions are saved. Parallel tree training/optimization may produce small numeric or equal-objective allocation differences across platforms.

The test targets are hidden and there is no official scoring script for prediction quality in the supplied material. No code can certify maximum unseen accuracy. Per-series demand errors are reported because pooled demand R² can look excellent simply from level differences between large and small series. Sparse Tech demand remains harder to predict, and the short final holdout cannot establish festival performance across every future peak.
