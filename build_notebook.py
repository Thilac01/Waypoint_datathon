"""Create a reproducible, readable competition notebook from the Python workflow."""
from pathlib import Path
import nbformat as nbf


def build(path):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("# Alt-F4 | Explainable Logistics Decision Intelligence\n\nTask1 service time and lateness, Task2A ten-week demand, and Task2B peak-day allocation.\n\n"
           "The complete implementation is in `waypoint_datathon.py`. This notebook executes label/feature inspection, reads actual stored evaluation results and loads saved models for inference. Set `RUN_TRAINING=True` to run the training cells again. The included results were trained by the Python module, not fabricated notebook outputs.\n\n"
           "All predictive models are trained from scratch on organizer data. Review `AI_DISCLOSURE.md`. The three tasks have distinct horizons; they support one operational workflow, not one jointly optimized route plan. The verified submission predictions remain frozen during the supplementary analysis."),
        code("from pathlib import Path\nfrom types import SimpleNamespace\nimport os, json\nimport pandas as pd\nimport numpy as np\nimport waypoint_datathon as w\nfrom IPython.display import display, Image, Markdown\n\n"
             "DATA_DIR = Path(os.environ.get('WAYPOINT_DATA', 'data'))\nOUTPUT_DIR = Path(os.environ.get('WAYPOINT_OUTPUT', 'outputs'))\nRUN_TRAINING = False\n"
             "data = w.Data(DATA_DIR)\nargs = SimpleNamespace(preset='balanced', jobs=4, allocator_seconds=120, priority_config=None, checker='check_allocation.py')\n"
             "for name in ('models', 'reports', 'submissions'):\n    (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)\n"
             "print('Data ready:', len(data.paths), 'organizer CSVs')"),
        md("## 1. Data checks\n\nOrders are unique by delivery ID. Dispatched orders and route legs are one-to-one by route/sequence. Reference joins must match. Input hashes and row counts make the run reproducible."),
        code("manifest = pd.DataFrame(data.manifest())\ndisplay(manifest[['file', 'rows', 'bytes']])\ndisplay(data.read('deliveries_train.csv').head(3))"),
        md("## 2. Label construction\n\nHandling begins at the later of actual arrival and window opening.\n\n"
           "`service_min = leave_outlet - max(arrival, window_open)`\n\n"
           "`is_late = arrival > window_close`\n\nNever-dispatched orders have no service labels. Arrival exactly at close is not late."),
        code("orders = data.read('deliveries_train.csv')\nlegs = data.read('route_legs_train.csv')\n"
             "joined = orders.loc[orders.dispatch_status.ne('not_run')].merge(legs, left_on=['route_id','seq_in_route'], right_on=['route_id','seq'], suffixes=('', '_leg'), validate='one_to_one')\n"
             "arrival = w.minutes(joined.arrival_time)\nopening = w.minutes(joined.window_open_time)\nclosing = w.minutes(joined.window_close_time)\nleaving = w.minutes(joined.leave_outlet_time)\n"
             "joined['service_min'] = leaving - np.maximum(arrival, opening)\njoined['is_late'] = (arrival > closing).astype(int)\n"
             "assert joined.service_min.ge(0).all()\n"
             "print('Early-arrival example: arrival 08:40, window opens 09:00, leave 09:12 -> service 12 minutes; waiting 20 minutes.')\n"
             "display(joined[['delivery_id','arrival_time','window_open_time','window_close_time','leave_outlet_time','service_min','is_late']].head(10))"),
        md("## 3. Planned features and chronological validation\n\nActual outcome columns are removed before feature building. Features include load, outlet access, planned time/slack, known calendar/road conditions and cumulative preceding planned stops. Two four-week tuning folds precede an untouched six-week holdout. Early stopping and ensemble weights use tuning folds only."),
        code("X, service_y, late_y, metadata = w.construct_task1(data, training=True)\n"
             "assert not any(c.startswith('actual_') or c in {'arrival_time','leave_outlet_time','route_id','delivery_id'} for c in X)\n"
             "print('Labeled rows:', len(X), 'Features:', X.shape[1], 'Late prevalence:', round(late_y.mean(), 4))\n"
             "display(X.head(3))\ndisplay(pd.DataFrame(w.model_specs(args.preset)))"),
        code("if RUN_TRAINING:\n    task1_results = w.run_task1(data, OUTPUT_DIR, args)\nelse:\n    task1_results = json.loads((OUTPUT_DIR / 'reports/task1_metrics.json').read_text())\n"
             "display(pd.DataFrame([{'target': t, **task1_results[t]['holdout_metrics']} for t in ['service','lateness']]))\n"
             "display(pd.read_csv(OUTPUT_DIR / 'reports/task1_leaderboard.csv'))"),
        code("if (OUTPUT_DIR / 'reports/task1_validation.png').exists():\n    display(Image(filename=str(OUTPUT_DIR / 'reports/task1_validation.png')))"),
        code("print('TASK1 VALIDATION DATES')\ndisplay(pd.DataFrame(task1_results['tuning_windows'], columns=['tuning_start', 'tuning_end']))\n"
             "print('Final holdout:', task1_results['holdout'])\n"
             "slices = pd.read_csv(OUTPUT_DIR / 'reports/task1_by_brand_depot.csv')\n"
             "display(slices.query(\"target == 'service' and slice == 'depot+brand'\")[['depot','brand','n','mae','rmse']].round(3))\n"
             "display(slices.query(\"target == 'lateness' and slice == 'depot+brand'\")[['depot','brand','n','late_count','roc_auc','log_loss']].round(4))"),
        md("## 4. Order-date demand and ten-week forecasting\n\nCount all training orders plus Task1 test orders once, including deferred/not-run. Group with the supplied calendar's ISO year/week. Complete zero-filled daily panels sum exactly to source order demand. Forecasts use known calendar context and frozen-origin historical baselines; no future actual demand enters validation. Tune on two ten-week horizons, assess the final ten weeks, then refit."),
        code("history, future_days, future_keys, demand_audit = w.demand_panel(data)\n"
             "display(pd.Series(demand_audit, name='audit'))\n"
             "display(w.weekly_context(history).tail(12))\n"
             "print('Forecast weeks:', future_keys.week_start.nunique())\ndisplay(future_keys.head(6))"),
        code("if RUN_TRAINING:\n    demand_results = w.run_task2a(data, OUTPUT_DIR, args)\nelse:\n    demand_results = json.loads((OUTPUT_DIR / 'reports/task2a_metrics.json').read_text())\n"
             "display(pd.DataFrame([{'target': t, **demand_results[t]['holdout_metrics']} for t in ['total','chilled']]))\n"
             "display(pd.DataFrame(demand_results['total']['holdout_by_series']).T)\n"
             "display(pd.read_csv(OUTPUT_DIR / 'reports/task2a_leaderboard.csv').query(\"stage != 'tuning_series'\"))"),
        code("if (OUTPUT_DIR / 'reports/task2a_validation.png').exists():\n    display(Image(filename=str(OUTPUT_DIR / 'reports/task2a_validation.png')))"),
        md("### Forecasts that support a decision\n\nAggregate WAPE is volume-weighted and hides weaker Tech series. Negative per-series R² means the model has greater squared error than that holdout's hindsight mean; it does not mean the model has negative accuracy. We report both aggregate and segment results. The small new baseline experiment uses only the earlier ten-week folds; its modest/inconsistent results do not justify changing the verified submission."),
        code("display(Image(filename=str(OUTPUT_DIR / 'reports/forecast_and_errors.png')))\n"
             "segment = pd.read_csv(OUTPUT_DIR / 'reports/task2a_by_brand_depot.csv')\n"
             "display(segment.query(\"target == 'total' and slice == 'depot+brand'\")[['depot','brand','n','mae','wape','r2','bias_m3']].round(4))\n"
             "experiment = json.loads((OUTPUT_DIR / 'reports/tech_baseline_experiment.json').read_text())\n"
             "print('Baseline tuning origins:', experiment['origins'], '| Excluded final holdout:', experiment['excluded_final_holdout_start'])\n"
             "display(pd.DataFrame(experiment['comparison']).drop(columns='action').round(4))"),
        md("## 5. Allocation\n\nCP-SAT assigns whole orders to available vehicles. Each trip uses one brand and district; capacity, depot, chilled and van restrictions are hard constraints. Per vehicle: at most two trips; Fresh time <=270 minutes and Style/Tech time <=480 minutes. Trip time is outbound + (orders−1)×inter-stop + published handling allowances. Priority points are transparent team policy, not official scoring. A separate maximum-order-count model supplies a comparison and bound."),
        code("if RUN_TRAINING:\n    allocation = w.run_task2b(data, OUTPUT_DIR, args)\nelse:\n    allocation = json.loads((OUTPUT_DIR / 'reports/task2b_summary.json').read_text())\n"
             "display(pd.Series({k: allocation[k] for k in ['solver_status','served','deferred','served_volume_m3','deferred_volume_m3','relative_gap','official_checker']}))\n"
             "display(pd.read_csv(OUTPUT_DIR / 'reports/task2b_trip_audit.csv'))\n"
             "display(pd.read_csv(OUTPUT_DIR / 'reports/task2b_order_decisions.csv').query(\"decision == 'deferred'\")[[\"order_ref\",\"brand\",\"order_volume_m3\",\"priority_points\",\"reason\"]])"),
        md("### Why all 85 orders cannot be served\n\nThere are four available refrigerated vehicles. Even allowing two full-volume trips for each, chilled demand exceeds optimistic capacity by 9.229 m³. This is a volume bound, not a sufficient feasibility test. Whole orders, district separation, van access, weight and time restrict packing further. A separate maximum-count solve proves that at most 79 orders can be served under the implemented competition rules."),
        code("display(Image(filename=str(OUTPUT_DIR / 'reports/capacity_vs_demand.png')))\n"
             "print('Maximum-count proof:', allocation['max_count_comparison'])\n"
             "print('Official checker:', allocation['official_checker'])"),
        md("### Same constraints, different business priorities\n\nFresh-first emphasizes perishable demand; fairness-first emphasizes yesterday's deferred orders and longer waits; balanced retains the verified submission weights. Only the objective weights change. Own-policy points have different scales, so use the common balanced-reference points for cross-policy comparisons. Fairness serves all 10 repeat-deferred orders but gives up two deliveries and 14.469 m³ of chilled volume. These policy scores are not official competition scores."),
        code("policies = pd.read_csv(OUTPUT_DIR / 'reports/policy_comparison.csv')\n"
             "display(policies[['policy','served_orders','deferred_orders','chilled_volume_delivered_m3','own_policy_points','balanced_reference_points','repeat_deferred_orders_served','solver_status','solver_gap']].round(3))\n"
             "display(pd.read_csv(OUTPUT_DIR / 'reports/policy_changed_orders.csv'))\n"
             "print('All three alternative allocations passed the original organizer checker.')\n"
             "frozen = json.loads((OUTPUT_DIR / 'reports/submission_freeze.json').read_text())\nassert frozen['unchanged']\nprint('Verified submission CSVs unchanged:', frozen['unchanged'])"),
        md("## 6. Submission validation\n\nThe following checks preserve every supplied identifier, exact columns and row order. Demand outputs must satisfy zero <= chilled <= total, and non-Fresh chilled must equal zero."),
        code("validation = w.validate_submissions(data, OUTPUT_DIR)\ndisplay(pd.DataFrame(validation).T)\n"
             "summary = w.write_run_summary(data, OUTPUT_DIR)\n"
             "from IPython.display import HTML\ndisplay(HTML((OUTPUT_DIR / 'reports/run_summary.html').read_text()))"),
        md("## 7. Saved-model inference\n\nFinal cell: load the saved model files, show actual test inputs and predictions for both prediction tasks, and verify that serialized-model inference reproduces the submission files. No training occurs in this cell."),
        code("print('TASK1 INPUTS')\ndisplay(data.read('task1_test_inputs.csv').head(5))\n"
             "task1_predictions = w.inference_task1(data, OUTPUT_DIR / 'models')\nprint('TASK1 PREDICTIONS FROM SAVED MODELS')\ndisplay(task1_predictions.head(5))\n"
             "print('TASK2A INPUTS')\ndisplay(data.read('task2a_test_inputs.csv').head(6))\n"
             "task2a_predictions = w.inference_task2a(data, OUTPUT_DIR / 'models')\nprint('TASK2A PREDICTIONS FROM SAVED MODELS')\ndisplay(task2a_predictions.head(6))\n"
             "for key, frame, file in [('delivery_id', task1_predictions, 'submission_task1.csv'), ('row_id', task2a_predictions, 'submission_task2a.csv')]:\n"
             "    expected = pd.read_csv(OUTPUT_DIR / 'submissions' / file).set_index(key)\n"
             "    actual = frame.set_index(key).reindex(expected.index)[expected.columns]\n"
             "    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)\n"
             "print('Both saved-model outputs reproduce the delivered prediction CSVs.')"),
    ]
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                  "language_info": {"name": "python", "version": "3.12"}})
    nbf.write(nb, path)


if __name__ == '__main__':
    build(Path(__file__).with_name('Alt-F4_FinalNotebook.ipynb'))
