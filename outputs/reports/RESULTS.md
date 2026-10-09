# Results from the included balanced run

These are chronological holdout results on the supplied synthetic data, not hidden-test or organizer scores. Model families, hyperparameters, blend weights and calibration were selected before the final holdouts. Submission models were then refitted on all permitted history.

| Output | Holdout metric | Result |
|---|---|---:|
| Service time | MAE (minutes) | 3.7524 |
| Service time | RMSE (minutes) | 6.0427 |
| Service time | R-squared | 0.8299 |
| Lateness | ROC-AUC | 0.9753 |
| Lateness | Log loss | 0.1323 |
| Lateness | Brier score | 0.0411 |
| Lateness | Accuracy at 0.5 | 94.50% |
| Total weekly demand | WAPE | 6.08% |
| Total weekly demand | RMSE (m3) | 28.6148 |
| Chilled weekly demand (Fresh only) | WAPE | 5.24% |
| Chilled weekly demand (Fresh only) | RMSE (m3) | 17.7373 |
| Peak-day allocation | Served / all orders | 79 / 85 |
| Peak-day allocation | Maximum count proved by solver | 79 |
| Official allocation checker | Result | PASSED |

Task1 holdout: 2026-01-04 through 2026-02-14. Demand holdout: the ten weeks beginning 2026-01-19. No random row splitting was used.

WAPE = sum of absolute errors / sum of observed volume. It is an error measure, not classification accuracy. ROC-AUC is a probability-ranking metric, not the fraction classified correctly.

## Selected Task1 blends

Service: [['catboost_depth6', 0.720378159106873], ['hist_leaves31', 0.279621840893127]].

Lateness: [['xgboost_depth5', 0.4748506140468604], ['catboost_depth6', 0.32896015798611294], ['lightgbm_leaves31', 0.19618922796702667]].

Calibration: none.

## Total-demand results by series

| Depot / brand | WAPE | MAE (m3) | R-squared |
|---|---:|---:|---:|
| Kandy / Fresh | 4.65% | 24.223 | -1.276 |
| Kandy / Style | 9.93% | 7.983 | 0.014 |
| Kandy / Tech | 24.67% | 5.137 | -0.707 |
| Peliyagoda / Fresh | 5.06% | 49.811 | -0.646 |
| Peliyagoda / Style | 7.70% | 11.182 | 0.018 |
| Peliyagoda / Tech | 29.83% | 10.341 | -0.791 |

The relatively large Tech errors and negative R-squared values in several individual series are material limitations. Pooled demand R-squared is dominated by between-series level differences; it is not evidence that week-to-week peaks are captured accurately. The Fresh forecasts underpredict parts of the final holdout. Do not present the pooled score alone or claim guaranteed highest accuracy.

## Validation evidence

- All 5,014 Task1, 60 Task2A and 85 Task2B rows have exactly the required columns and original identifiers/order.
- Saved-model replay reproduces both prediction CSVs within 1e-7 absolute tolerance.
- Five focused checks pass: waiting-aware service and strict-close labels, actual-outcome feature isolation, demand counting/date assignment, submission/constraint validation including deliberate corruptions, and saved-model replay.
- The organizer's unmodified feasibility checker passes. Both allocation objectives have OPTIMAL status and zero reported gap. Optimality applies to the explicitly modelled published rules and objective, not a real-world routing problem with additional constraints.
- An independent maximum-count objective proves at least six deferrals are collectively required. Only S1-078 is individually infeasible on every available vehicle. Which other orders are deferred is an allocation-policy choice under shared constraints.

## What remains for the team

The final notebook is named `Alt-F4_FinalNotebook.ipynb`. Review the implementation and AI disclosure. The package includes a 4-minute-10-second demo review file with synthetic narration over actual executed notebook/report excerpts; it is a rendered replay, not a live human recording. Review it or record the timed script yourselves, upload the reviewed video as unlisted, and submit its actual link. No competition submission or video upload has been performed. The supplied booklet does not provide exact prediction-scoring metrics, so no official score or ranking can be certified.

## Added decision evidence

Fresh-first and Balanced each serve 79 orders, delivering 140.723 m³ chilled and serving 9 of 10 repeat-deferred orders. Fairness-first serves 77 orders, delivering 126.254 m³ chilled while serving all 10 repeat-deferred orders. The policies change six order decisions. Every solution is optimal with 0% gap and passes the original feasibility checker; all hard constraints are unchanged. See `POLICY_SENSITIVITY.md` for weights and common-reference scores.

Brand/depot service and lateness metrics and all demand-series errors appear in `EVALUATION_BY_SEGMENT.md`. Nine simple Tech baselines were tested only on the two earlier chronological validation windows. Kandy's best bias-corrected mean improves earlier RMSE by 0.90%; Peliyagoda's best new baseline is 13.24% worse than the verified ridge model. These modest/inconsistent changes do not establish improved unseen performance. The verified submissions and models remain unchanged.
