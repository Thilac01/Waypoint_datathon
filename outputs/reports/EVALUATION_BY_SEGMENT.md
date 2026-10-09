# Model evaluation by brand and depot

These slices reuse the original frozen-model holdouts. No model is selected using these final-period results.

## Service time and lateness

| Depot | Brand | Orders | Service MAE (min) | Lateness AUC | Log loss |
|---|---|---:|---:|---:|---:|
| Kandy | Fresh | 1860 | 3.331 | 0.959 | 0.181 |
| Kandy | Style | 54 | 9.220 | 0.962 | 0.060 |
| Kandy | Tech | 33 | 15.267 | 0.968 | 0.094 |
| Peliyagoda | Fresh | 2929 | 3.376 | 0.983 | 0.108 |
| Peliyagoda | Style | 96 | 10.445 | 0.991 | 0.055 |
| Peliyagoda | Tech | 63 | 12.805 | 1.000 | 0.022 |

## Weekly total demand

| Depot | Brand | Weeks | MAE (m3) | WAPE | R-squared | Mean prediction bias (m3) |
|---|---|---:|---:|---:|---:|---:|
| Kandy | Fresh | 10 | 24.223 | 4.65% | -1.276 | -23.784 |
| Kandy | Style | 10 | 7.983 | 9.93% | 0.014 | -0.574 |
| Kandy | Tech | 10 | 5.137 | 24.67% | -0.707 | 4.391 |
| Peliyagoda | Fresh | 10 | 49.811 | 5.06% | -0.646 | -49.811 |
| Peliyagoda | Style | 10 | 11.182 | 7.70% | 0.018 | -7.313 |
| Peliyagoda | Tech | 10 | 10.341 | 29.83% | -0.791 | -8.194 |

Tech's intermittent low-volume demand has larger relative errors. Several negative per-series R-squared values mean that, on those ten weeks, a constant equal to that holdout's actual mean would have lower squared error. That hindsight mean is not a deployable forecast. The pooled R-squared is dominated by between-series level differences. Report WAPE and the full per-series table instead of presenting pooled R-squared as proof of peak prediction.
