"""Reproducible decision evidence without altering the verified submissions.

python decision_intelligence.py --data data --output outputs --task all
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import waypoint_datathon as w

POLICIES = {
    "Fresh-first": {"base": 100, "Fresh": 1200, "Style": 0, "Tech": 0,
                    "chilled_bonus": 400, "deferred_yesterday_bonus": 0, "per_day_unserved": 0, "days_cap": 7},
    "Fairness-first": {"base": 100, "Fresh": 0, "Style": 0, "Tech": 0,
                       "chilled_bonus": 0, "deferred_yesterday_bonus": 2000, "per_day_unserved": 250, "days_cap": 7},
    "Balanced": w.DEFAULT_PRIORITY.copy(),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_check(data, checker_path, allocation_path):
    spec = importlib.util.spec_from_file_location("official_policy_checker", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker._find = lambda name: str(data.paths[name])
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        result = checker.check(str(allocation_path))
    w.require(result == 0, capture.getvalue())
    return capture.getvalue().strip()


def policy_sensitivity(data, out, checker_path, seconds=90, jobs=3):
    folder = out / "reports/policies"
    folder.mkdir(parents=True, exist_ok=True)
    orders, available, _, _, _ = w.allocation_references(data)
    points_reference = {r.order_ref: w.priority_points(r, POLICIES["Balanced"]) for r in orders.itertuples()}
    rows, decisions, solver_reports = [], [], {}
    for name, policy in POLICIES.items():
        allocation, solver = w.solve_allocation(data, seconds, jobs, policy, "priority")
        trips = w.validate_allocation(data, allocation)
        slug = name.lower()
        path = folder / f"allocation_{slug}.csv"
        allocation.to_csv(path, index=False)
        trips.to_csv(folder / f"trip_audit_{slug}.csv", index=False)
        checker = official_check(data, checker_path, path)
        (folder / f"checker_{slug}.txt").write_text(checker)
        m = orders.merge(allocation.drop(columns="outlet_id"), on=["scenario", "order_ref"], validate="one_to_one")
        served = m.decision.eq("served")
        chilled = m.temp_requirement.eq("chilled")
        previous = m.deferred_yesterday.eq(1)
        rows.append({"policy": name, "served_orders": int(served.sum()), "deferred_orders": int((~served).sum()),
                     "chilled_volume_delivered_m3": float(m.loc[served & chilled, "order_volume_m3"].sum()),
                     "total_volume_delivered_m3": float(m.loc[served, "order_volume_m3"].sum()),
                     "own_policy_points": solver["priority_points_served"],
                     "balanced_reference_points": int(sum(points_reference[k] for k in m.loc[served, "order_ref"])),
                     "repeat_deferred_orders_served": int((served & previous).sum()),
                     "repeat_deferred_orders_total": int(previous.sum()),
                     "max_days_waiting_among_deferred": int(m.loc[~served, "days_since_last_served"].max()),
                     "solver_status": solver["solver_status"], "solver_gap": solver["relative_gap"],
                     "official_checker": "PASSED", "deferred_order_refs": ", ".join(m.loc[~served, "order_ref"]),
                     "seconds": solver["seconds"]})
        decisions.append(m[["order_ref", "brand", "district", "temp_requirement", "order_volume_m3", "deferred_yesterday", "days_since_last_served", "decision"]].assign(policy=name))
        solver_reports[name] = {"weights": policy, "solver": solver, "official_checker": checker}
        print(name, rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "reports/policy_comparison.csv", index=False)
    detail = pd.concat(decisions, ignore_index=True)
    detail.to_csv(out / "reports/policy_order_decisions.csv", index=False)
    wide = detail.pivot(index="order_ref", columns="policy", values="decision")
    changed = wide[wide.nunique(axis=1) > 1]
    changed.to_csv(out / "reports/policy_changed_orders.csv")
    reefers = available[available.temp.eq("reefer")]
    demand = float(orders.loc[orders.temp_requirement.eq("chilled"), "order_volume_m3"].sum())
    optimistic = float(2 * reefers.volume_cap_m3.sum())
    audit = {"policies": solver_reports, "changed_order_count": len(changed),
             "same_served_sets": bool(len(changed) == 0), "chilled_demand_m3": demand,
             "optimistic_refrigerated_capacity_m3": optimistic, "optimistic_shortfall_m3": demand - optimistic,
             "available_vehicles": len(available), "available_reefers": len(reefers),
             "available_refrigerated_vans": int(reefers.type.eq("van").sum()),
             "constraint_engine_sha256": hashlib.sha256(inspect.getsource(w.solve_allocation).encode()).hexdigest(),
             "method": "Same CP-SAT function and hard rules for each run; only integer benefit weights change.",
             "comparison_rule": "Own-policy points have different scales. Compare balanced-reference points across rows.",
             "selection": "The verified original Balanced submission is retained; these are supplementary policy experiments."}
    w.write_json(out / "reports/policy_sensitivity.json", audit)
    interpretation = ("All three policies serve the same set of orders in this scenario. The tested weight changes do not alter these decisions. "
                      "This is robustness within the tested policies, not a claim that all possible policies would agree."
                      if len(changed) == 0 else
                      f"The policies change the served/deferred decision for {len(changed)} orders. The order-level differences are recorded in policy_changed_orders.csv.")
    text = "# Allocation policy sensitivity\n\n" + audit["method"] + "\n\n" + audit["comparison_rule"] + "\n\n"
    text += "| Policy | Served | Chilled delivered (m3) | Own points | Common balanced points | Deferred | Status |\n|---|---:|---:|---:|---:|---:|---|\n"
    for r in table.itertuples():
        text += f"| {r.policy} | {r.served_orders} | {r.chilled_volume_delivered_m3:.3f} | {r.own_policy_points} | {r.balanced_reference_points} | {r.deferred_orders} | {r.solver_status} |\n"
    text += f"\n{interpretation}\n\nThe verified Balanced allocation remains the submission. Every candidate passes the organizer checker. "
    text += f"The cold-chain demand of {demand:.3f} m3 exceeds the optimistic two-trip refrigeration capacity of {optimistic:.3f} m3 by {demand - optimistic:.3f} m3. "
    text += "That simple bound already rules out serving all chilled volume. Whole-order packing, district separation, weight, van access and time impose additional limits. "
    text += "The 79-order maximum-count proof is reported separately in task2b_summary.json. Optimality covers the published competition model.\n"
    (out / "reports/POLICY_SENSITIVITY.md").write_text(text)
    return table


def evaluation_slices(out):
    rows = []
    for target in ["service", "lateness"]:
        df = pd.read_csv(out / f"reports/task1_{target}_holdout.csv")
        for dimensions in [["brand"], ["depot"], ["depot", "brand"]]:
            for key, g in df.groupby(dimensions):
                if not isinstance(key, tuple):
                    key = (key,)
                m = w.metrics(g.actual, g.prediction, classification=target == "lateness")
                rows.append({"target": target, "slice": "+".join(dimensions),
                             **dict(zip(dimensions, key)), "n": len(g),
                             "late_count": int(g.actual.sum()) if target == "lateness" else None, **m})
    task1 = pd.DataFrame(rows)
    task1.to_csv(out / "reports/task1_by_brand_depot.csv", index=False)
    forecast = pd.read_csv(out / "reports/task2a_holdout.csv")
    rows = []
    for target, col in [("total", "pred_total_volume_m3"), ("chilled", "pred_chilled_volume_m3")]:
        df = forecast[forecast.brand.eq("Fresh")] if target == "chilled" else forecast
        for dimensions in [["brand"], ["depot"], ["depot", "brand"]]:
            for key, g in df.groupby(dimensions):
                if not isinstance(key, tuple):
                    key = (key,)
                rows.append({"target": target, "slice": "+".join(dimensions), **dict(zip(dimensions, key)), "n": len(g),
                             "bias_m3": float((g[col] - g[target]).mean()), **w.metrics(g[target], g[col])})
    task2 = pd.DataFrame(rows)
    task2.to_csv(out / "reports/task2a_by_brand_depot.csv", index=False)
    text = "# Model evaluation by brand and depot\n\nThese slices reuse the original frozen-model holdouts. No model is selected using these final-period results.\n\n"
    text += "## Service time and lateness\n\n| Depot | Brand | Orders | Service MAE (min) | Lateness AUC | Log loss |\n|---|---|---:|---:|---:|---:|\n"
    s = task1[(task1.target == "service") & (task1.slice == "depot+brand")]
    for r in s.itertuples():
        c = task1[(task1.target == "lateness") & (task1.slice == "depot+brand") & (task1.depot == r.depot) & (task1.brand == r.brand)].iloc[0]
        auc = f"{c.roc_auc:.3f}" if pd.notna(c.roc_auc) else "n/a: one class"
        text += f"| {r.depot} | {r.brand} | {r.n} | {r.mae:.3f} | {auc} | {c.log_loss:.3f} |\n"
    text += "\n## Weekly total demand\n\n| Depot | Brand | Weeks | MAE (m3) | WAPE | R-squared | Mean prediction bias (m3) |\n|---|---|---:|---:|---:|---:|---:|\n"
    for r in task2[(task2.target == "total") & (task2.slice == "depot+brand")].itertuples():
        text += f"| {r.depot} | {r.brand} | {r.n} | {r.mae:.3f} | {r.wape:.2%} | {r.r2:.3f} | {r.bias_m3:.3f} |\n"
    text += "\nTech's intermittent low-volume demand has larger relative errors. Several negative per-series R-squared values mean that, on those ten weeks, a constant equal to that holdout's actual mean would have lower squared error. That hindsight mean is not a deployable forecast. The pooled R-squared is dominated by between-series level differences. Report WAPE and the full per-series table instead of presenting pooled R-squared as proof of peak prediction.\n"
    (out / "reports/EVALUATION_BY_SEGMENT.md").write_text(text)
    return task1, task2


def tech_baseline_experiment(data, out):
    """Earlier chronological validation only. Keep submissions fixed in this sprint."""
    history, _, _, _ = w.demand_panel(data)
    weekly = w.weekly_context(history)
    # Final-holdout targets never enter fitting, bias correction or ranking.
    weeks = sorted(history.week_start.unique())
    stop = pd.Timestamp(weeks[-10])
    weekly = weekly[(weekly.week_start < stop) & weekly.brand.eq("Tech")].copy()
    origins = [pd.Timestamp(weeks[-30]), pd.Timestamp(weeks[-20])]
    candidates = [("mean_4", "mean", 4), ("mean_13", "mean", 13), ("mean_26", "mean", 26),
                  ("mean_52", "mean", 52), ("median_13", "median", 13),
                  ("ewma_8", "ewma", 8), ("ewma_13", "ewma", 13),
                  ("mean_13_prior_bias", "bias", 13), ("mean_26_prior_bias", "bias", 26)]
    records = []
    for depot, series in weekly.groupby("depot"):
        series = series.sort_values("week_start")
        for origin in origins:
            tr = series[series.week_start < origin]
            va = series[(series.week_start >= origin) & (series.week_start < origin + pd.Timedelta(weeks=10))]
            y = tr.total.to_numpy()
            for name, family, window in candidates:
                if family == "median":
                    p = np.median(y[-window:])
                elif family == "ewma":
                    p = pd.Series(y).ewm(span=window, adjust=False).mean().iloc[-1]
                else:
                    p = np.mean(y[-window:])
                    if family == "bias":
                        # Residual correction from historical one-step forecasts only.
                        residual = [y[t] - np.mean(y[max(0, t-window):t]) for t in range(max(window, len(y)-13), len(y))]
                        p += float(np.mean(residual)) if residual else 0
                for r in va.itertuples():
                    records.append({"depot": depot, "candidate": name, "origin": origin.date().isoformat(),
                                    "week_start": r.week_start.date().isoformat(), "actual": r.total, "prediction": max(0, float(p))})
    values = pd.DataFrame(records)
    rows = []
    for (depot, name), g in values.groupby(["depot", "candidate"]):
        rows.append({"depot": depot, "candidate": name, "weeks": len(g), **w.metrics(g.actual, g.prediction)})
    board = pd.DataFrame(rows).sort_values(["depot", "rmse"])
    original = pd.read_csv(out / "reports/task2a_leaderboard.csv")
    prior_models = original[(original.target == "total") & (original.stage == "tuning_series") & original.brand.eq("Tech")]
    # Compare against the originally selected model, using its already saved earlier validation scores.
    current = json.loads((out / "reports/task2a_metrics.json").read_text())["total"]["selection"]
    comparisons = []
    for depot in sorted(board.depot.unique()):
        best = board[board.depot.eq(depot)].iloc[0]
        terms = current[depot + ":Tech"]
        w.require(len(terms) == 1, "Tech reference changed: recompute blend OOF before comparing")
        reference = prior_models[(prior_models.depot == depot) & (prior_models.model == terms[0][0])].iloc[0]
        comparisons.append({"depot": depot, "reference_model": terms[0][0], "reference_tuning_rmse": float(reference.rmse),
                            "best_new_baseline": best.candidate, "baseline_tuning_rmse": float(best.rmse),
                            "relative_rmse_improvement": float(1 - best.rmse / reference.rmse),
                            "action": "Retain verified submission; exploratory earlier-validation evidence only."})
    board.to_csv(out / "reports/tech_baseline_validation.csv", index=False)
    values.to_csv(out / "reports/tech_baseline_oof_predictions.csv", index=False)
    w.write_json(out / "reports/tech_baseline_experiment.json", {"origins": [str(x.date()) for x in origins],
        "excluded_final_holdout_start": str(stop.date()), "comparison": comparisons,
        "protocol": "Nine explicitly defined simple baselines; expanding training, frozen ten-week forecasts. Bias uses older one-step residuals. Final holdout is excluded. No submission prediction or saved model is changed."})
    text = "# Small Tech forecasting experiment\n\nThe final holdout (starting " + str(stop.date()) + ") is excluded from this experiment. Nine simple candidates use the two earlier ten-week validation windows, matching the original tuning origins. Bias correction uses only one-step residuals available before each origin.\n\n"
    text += "| Depot | Existing model | Earlier RMSE | Best simple baseline | Earlier RMSE | Relative change |\n|---|---|---:|---|---:|---:|\n"
    for r in comparisons:
        text += f"| {r['depot']} | {r['reference_model']} | {r['reference_tuning_rmse']:.3f} | {r['best_new_baseline']} | {r['baseline_tuning_rmse']:.3f} | {r['relative_rmse_improvement']:.2%} improvement |\n"
    text += "\nThe verified submission remains frozen for this deadline sprint. This experiment documents a targeted next step without presenting repeated tuning against the known final holdout as independent evidence. It does not establish improved unseen performance.\n"
    (out / "reports/TECH_BASELINE_EXPERIMENT.md").write_text(text)
    print(pd.DataFrame(comparisons).to_string(index=False))
    return board


def business_evidence(data, out):
    one = pd.read_csv(out / "submissions/submission_task1.csv")
    plan = data.read("task1_test_inputs.csv").merge(one, on="delivery_id", validate="one_to_one")
    task1 = {"planned_orders": len(plan), "flagged_at_0_5": int(plan.pred_late_prob.ge(.5).sum()),
             "expected_late_count": float(plan.pred_late_prob.sum()),
             "model_expected_total_handling_hours": float(plan.pred_service_min.sum() / 60),
             "meaning": "Across the full six-week test horizon. These are predictions, not observed outcomes; summed service hours exclude travel and are not driver requirements."}
    two = pd.read_csv(out / "submissions/submission_task2a.csv").merge(data.read("task2a_test_inputs.csv"), on="row_id", validate="one_to_one")
    weekly = two.groupby(["iso_year", "iso_week"], as_index=False)[["pred_total_volume_m3", "pred_chilled_volume_m3"]].sum()
    peak = weekly.loc[weekly.pred_total_volume_m3.idxmax()]
    w.write_json(out / "reports/decision_evidence.json", {"task1": task1, "forecast_peak_week": peak.to_dict(),
        "forecast_total_m3": float(two.pred_total_volume_m3.sum()), "forecast_chilled_m3": float(two.pred_chilled_volume_m3.sum()),
        "scope": "The Task1 test horizon, Task2A future horizon and Task2B supplied scenario are distinct. Their outputs support one operating workflow but are not one jointly solved route plan."})
    weekly.to_csv(out / "reports/ten_week_network_forecast.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="outputs")
    p.add_argument("--checker", default="check_allocation.py")
    p.add_argument("--task", choices=["all", "policies", "evaluation", "tech"], default="all")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--seconds", type=float, default=90)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    data, out = w.Data(args.data), Path(args.output)
    before = {f.name: digest(f) for f in (out / "submissions").glob("*.csv")}
    if args.task in {"all", "policies"}:
        policy_sensitivity(data, out, args.checker, args.seconds, args.jobs)
    if args.task in {"all", "evaluation"}:
        evaluation_slices(out)
    if args.task in {"all", "tech"}:
        tech_baseline_experiment(data, out)
    business_evidence(data, out)
    after = {f.name: digest(f) for f in (out / "submissions").glob("*.csv")}
    w.require(before == after, "Supplementary analysis unexpectedly changed submission files")
    w.write_json(out / "reports/submission_freeze.json", {"before": before, "after": after, "unchanged": True})
    print("Supplementary evidence complete. All three verified submissions are unchanged.")


if __name__ == "__main__":
    main()
