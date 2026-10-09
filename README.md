# Waypoint Datathon Python solution

A complete, local Python workflow for the three tasks in the Tech-Triathlon 2026 Challenge Booklet. The supplied organizer data stays in your own `data/` folder. No API key, network request, pretrained model or AutoML platform is used by the pipeline.

## Start here

Use Python 3.11 or 3.12 (the included results were produced on Python 3.12). Extract this package and put your original extracted `data` folder beside `waypoint_datathon.py`. Keep its `General Data`, `Training Data`, `Test Data` and `Submission Templates` subfolders. The program finds the 17 CSV files recursively; it rejects missing or ambiguous filenames.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or on Linux/macOS with `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
python waypoint_datathon.py --data "data" --output "outputs" --preset balanced --checker "check_allocation.py"
```

The balanced run compares 8 configurations for each Task1 target and 10 candidates for each demand target. It trains, evaluates, refits, saves the models, writes all three submissions, checks allocation with the organizer's unmodified checker, and validates the final files. Runtime depends on your CPU; boosting comparisons are the main cost. Progress goes to the terminal, `outputs/run.log`, and incremental leaderboards.

For a wider, slower fixed hyperparameter comparison (13 configurations per Task1 target):

```bash
python waypoint_datathon.py --data "data" --output "thorough_outputs" --preset thorough --jobs 4 --allocator-seconds 180 --checker "check_allocation.py"
```

This is an explicit set of candidate models, not an exhaustive search of all possible algorithms. More computation does not guarantee a better unseen score. The booklet does not publish an exact prediction-scoring formula. Service selection uses RMSE, probability selection uses log loss, and each demand series is selected by weekly RMSE; all reports also include alternative metrics. Holdout results are local estimates, not organizer scores or a guaranteed competition ranking.

For a quick first run use `--preset quick`. To run one task add `--task task1`, `--task task2a`, or `--task task2b`. Use the same output folder when completing the three tasks separately. Avoid simultaneous writers to the same task's output files.

## Included trained results

The `outputs/` directory in this download contains the completed balanced run: models, predictions, metrics, validation figures and the allocation policy. You can inspect or run inference immediately after installing the dependencies and supplying your original data.

```bash
python waypoint_datathon.py --data "data" --output "outputs" --task predict
python waypoint_datathon.py --data "data" --output "outputs" --task validate
python test_solution.py --data "data" --output "outputs" -v
```

The `predict` command reloads the saved models in a fresh process and prints actual input rows alongside their predictions. It does not retrain. The `validate` command checks exact template columns, identifiers, row order, finite values, probabilities, chilled/total consistency, and allocation constraints. To run the original checker directly, leave `check_allocation.py` beside `data/`:

```bash
python check_allocation.py outputs/submissions/submission_task2b.csv
```

Model files are Python joblib objects. Load these supplied files or your own outputs; use the recorded package versions in `outputs/reports/environment.json` when reproducing saved models. `requirements.txt` allows compatible installation ranges, while `requirements-tested.txt` records this run's exact principal package versions.

## What each task does

**Task1:** joins each dispatched order to its matching `(route_id, seq_in_route)` leg, excludes orders that never ran, and constructs `service = leave - max(arrival, window_open)` and `late = arrival > window_close`. Actual outcomes are removed before feature engineering. Features cover loads, access, planned timing, calendar, traffic, road disruption and preceding planned stops. Candidate families are CatBoost, LightGBM, XGBoost, histogram gradient boosting, Extra Trees, and linear/logistic baselines; thorough mode also includes Random Forest. Boosting configurations vary depth, leaves and regularization.

Two expanding chronological four-week tuning folds choose models, boosting rounds, ensemble weights and any probability calibration. A later untouched six-week holdout assesses the frozen choices. Final models are refitted on all labeled history. Routes cannot cross validation boundaries. Calibration is selected only inside the tuning period; the final holdout remains independent.

**Task2A:** uses every historical order plus Task1 test input orders once, including `deferred` and `not_run` rows. Demand is assigned by `order_date` and the calendar's ISO year/week. The complete 117-week panel fills absent orders with zero. Calendar ridge regressions, daily/weekly boosted models, rolling-mean baselines and a 52-week seasonal baseline forecast complete ten-week horizons. Models and convex blends are selected separately per depot/brand/target on two ten-week backtests. A final ten-week holdout is assessed before refitting. Future target lags are never filled with future actuals. Only Fresh may have chilled volume; chilled is constrained between zero and total.

**Task2B:** a CP-SAT integer model assigns whole orders to available vehicles and at most two trips per vehicle. It enforces matching depot, one brand/district per trip, refrigeration, van-only access, both capacity limits, and separate daily 270-minute Fresh / 480-minute Style-and-Tech budgets. Planned trip times use the supplied district and service-allowance tables, with no extra return journey. It compares the documented priority objective with a maximum-order-count objective and reports solver bounds/status. Priority weights are editable in `priority_config.json` and passed with `--priority-config priority_config.json`.

This Task2B model follows the published task and checker. It does not infer unprovided remaining weekly fuel, build detailed stop sequences, or claim individual arrival-window feasibility beyond the published aggregate budgets. Global Hackathon routing requirements are a different scope.

## Files to inspect

- `waypoint_datathon.py`: complete Python implementation, including reusable training and inference functions.
- `Alt+ F4_FinalNotebook.ipynb`: executed walkthrough of label construction, features, model configurations, saved evaluation results, allocation and saved-model inference. Training cells are available behind `RUN_TRAINING=True`; the included models were trained by the Python module.
- `outputs/submissions/submission_task1.csv`: 5,014 predictions, original template order.
- `outputs/submissions/submission_task2a.csv`: 60 weekly volume forecasts.
- `outputs/submissions/submission_task2b.csv`: 85 order decisions.
- `outputs/models/`: four saved ensembles (Task1 service/lateness and Task2A total/chilled).
- `outputs/reports/RESULTS.md`: actual holdout results and limitations.
- `outputs/reports/task1_leaderboard.csv`, `task2a_leaderboard.csv`: all completed comparisons, with tuning and holdout explicitly separated.
- `outputs/reports/task2b_trip_audit.csv`: trip-by-trip capacity/time calculations.
- `outputs/reports/task2b_order_decisions.csv`: priority, decision and reason for each order.
- `outputs/reports/task2b_policy.md`: short allocation policy, resource calculations and opportunity cost.
- `outputs/reports/official_checker.txt`: organizer checker result.
- `PREPROCESSING.md`, `ARCHITECTURE.md`, `AI_DISCLOSURE.md`, `DEMO_OUTLINE.md`: supporting competition documentation.

Original datasets are not bundled in this download. Preserve their competition-only use and confidentiality. For submission, replace `Alt+ F4` in the notebook filename, review the AI disclosure, and record the required unlisted 3–5 minute demo. The code does not record/upload a video or submit on your behalf. Place the reviewed deliverables in your team folder and zip it as `Alt+ F4_Datathon.zip` as required by the booklet.

The included notebook's 12 code cells were executed in order through in-process IPython because the build environment does not permit a network Jupyter kernel. All displayed tables, figures and predictions are actual outputs. Open it normally in Jupyter, or rerun without a kernel server using `python execute_notebook.py Alt+ F4_FinalNotebook.ipynb` after placing `data/` and `outputs/` beside it. `python build_notebook.py` regenerates the notebook without saved outputs.

## Primary references

- Supplied Challenge Booklet: Datathon tasks pp. 15–21; rules/deliverables pp. 22–23; data definitions pp. 24–31.
- Supplied `check_allocation.py`: feasibility rules and planned trip-time calculation.
- CatBoost training interface: https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit
- scikit-learn histogram boosting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- OR-Tools CP-SAT: https://developers.google.com/optimization/cp/cp_solver
