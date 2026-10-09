# Small Tech forecasting experiment

The final holdout (starting 2026-01-19) is excluded from this experiment. Nine simple candidates use the two earlier ten-week validation windows, matching the original tuning origins. Bias correction uses only one-step residuals available before each origin.

| Depot | Existing model | Earlier RMSE | Best simple baseline | Earlier RMSE | Relative change |
|---|---|---:|---|---:|---:|
| Kandy | mean_13 | 10.808 | mean_13_prior_bias | 10.710 | 0.90% improvement |
| Peliyagoda | ridge_1 | 8.178 | mean_26 | 9.261 | -13.24% improvement |

The verified submission remains frozen for this deadline sprint. This experiment documents a targeted next step without presenting repeated tuning against the known final holdout as independent evidence. It does not establish improved unseen performance.
