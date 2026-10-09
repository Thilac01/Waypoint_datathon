#!/usr/bin/env python3
"""Waypoint Datathon: explicit model comparison, forecasting and fleet allocation.

Run `python waypoint_datathon.py --help`. All models are trained from scratch.
No AutoML service, pretrained model, API key or network access is used by this code.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

LOG = logging.getLogger("waypoint")
SEED = 2026
REQUIRED_FILES = [
    "deliveries_train.csv", "route_legs_train.csv", "task1_test_inputs.csv",
    "route_legs_test.csv", "task2a_test_inputs.csv", "task2b_peak_day_scenarios.csv",
    "task2b_peak_day_fleet.csv", "outlets.csv", "vehicles.csv", "calendar.csv",
    "district_travel.csv", "service_allowance.csv", "traffic_speed.csv", "road_conditions.csv",
    "submission_task1.csv", "submission_task2a.csv", "submission_task2b.csv",
]


def require(ok, message):
    """Unlike assert, validation remains active under python -O."""
    if not bool(ok):
        raise ValueError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


class Data:
    """Resolve the organizer's filenames, regardless of the enclosing folder name."""
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        require(self.root.is_dir(), f"Data folder does not exist: {self.root}")
        self.paths = {}
        for name in REQUIRED_FILES:
            matches = list(self.root.rglob(name))
            require(len(matches) == 1,
                    f"Expected one {name} under {self.root}; found {len(matches)}. "
                    "Point --data at the extracted original data folder, not an output folder.")
            self.paths[name] = matches[0]
        self.cache = {}

    def read(self, name):
        if name not in self.cache:
            self.cache[name] = pd.read_csv(self.paths[name])
        return self.cache[name].copy()

    def manifest(self):
        return [{"file": str(p.relative_to(self.root)), "bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                 "rows": len(self.read(n))} for n, p in self.paths.items()]


def minutes(s):
    parts = s.astype("string").str.extract(r"^(\d{2}):(\d{2})$").astype(float)
    require(parts.notna().all().all(), f"Missing or malformed HH:MM clock values in {s.name}")
    require(((parts[0] < 24) & (parts[1] < 60)).all(), f"Invalid clock values in {s.name}")
    return parts[0] * 60 + parts[1]


def add_reference(frame, reference, keys, columns):
    require(not reference.duplicated(keys).any(), f"Duplicate reference keys: {keys}")
    n = len(frame)
    result = frame.merge(reference[keys + columns], on=keys, how="left", validate="many_to_one", indicator=True)
    require(len(result) == n and result._merge.eq("both").all(), f"Unmatched join on {keys}")
    return result.drop(columns="_merge")


def construct_task1(data, training=True):
    """Join each dispatched order to exactly one leg; keep actuals out of features."""
    orders = data.read("deliveries_train.csv" if training else "task1_test_inputs.csv")
    legs = data.read("route_legs_train.csv" if training else "route_legs_test.csv")
    require(not orders.delivery_id.duplicated().any(), "Duplicate order IDs")
    if training:
        require(orders.loc[orders.dispatch_status.eq("not_run"), "route_id"].isna().all(),
                "A not_run order unexpectedly has a route")
        orders = orders[orders.dispatch_status.ne("not_run")].copy()
    require(orders.route_id.notna().all(), "Dispatched order missing route")
    orders["seq_in_route"] = orders.seq_in_route.astype(int)
    legs = legs.rename(columns={"seq": "seq_in_route"})
    keys = ["route_id", "seq_in_route"]
    require(not orders.duplicated(keys).any() and not legs.duplicated(keys).any(), "Nonunique route/sequence")
    d = orders.merge(legs, on=keys, how="left", suffixes=("", "_leg"), validate="one_to_one", indicator=True)
    require(d._merge.eq("both").all() and len(d) == len(legs), "Order/leg mapping is incomplete")
    require(d.outlet_id.eq(d.to_outlet).all(), "Route destination does not match order outlet")
    require(d.dispatch_date.eq(d.date).all(), "Route date differs from dispatch date")
    for col in ["depot", "vehicle_id", "vehicle_type", "vehicle_temp", "brand", "district", "planned_arrival_time"]:
        require(d[col].eq(d[col + "_leg"]).all(), f"Order/leg mismatch: {col}")
    d = d.drop(columns=[c for c in d if c.endswith("_leg")] + ["_merge"])
    service = late = None
    if training:
        arrival, leave = minutes(d.arrival_time), minutes(d.leave_outlet_time)
        window_open, window_close = minutes(d.window_open_time), minutes(d.window_close_time)
        # The supplied records have same-day, non-overnight windows and completion.
        # Fail loudly for a different convention rather than guessing a midnight rollover.
        require((window_close >= window_open).all(), "Overnight windows require explicit date handling")
        service = (leave - np.maximum(arrival, window_open)).to_numpy(float)
        require(np.isfinite(service).all() and (service >= 0).all(), "Invalid service labels; inspect clock rollover")
        late = (arrival > window_close).to_numpy(int)  # Equal to close is NOT late.
    meta = d[["delivery_id", "route_id", "dispatch_date", "brand", "depot", "outlet_id"]].copy()
    meta["dispatch_date"] = pd.to_datetime(meta.dispatch_date)
    # Explicitly remove observed outcomes BEFORE any feature transformation/grouping.
    d = d.drop(columns=[c for c in d if c.startswith("actual_") or c in {"arrival_time", "leave_outlet_time"}])
    x = task1_features(data, d)
    require(not any(c.startswith("actual_") or c in {"arrival_time", "leave_outlet_time", "route_id", "delivery_id"}
                    for c in x), "Outcome/identifier leakage")
    return x, service, late, meta


def task1_features(data, d):
    d = d.copy()
    d["_position"] = np.arange(len(d))
    d = add_reference(d, data.read("outlets.csv"), ["outlet_id"], ["dock_type", "parking_constraint"])
    d = add_reference(d, data.read("vehicles.csv"), ["vehicle_id"],
                      ["weight_cap_kg", "volume_cap_m3", "km_per_l", "weekly_fuel_quota_l"])
    travel = data.read("district_travel.csv")
    d = add_reference(d, travel, ["district"], [c for c in travel if c not in {"district", "depot"}])
    cal = data.read("calendar.csv").rename(columns={"date": "dispatch_date"})
    d = d.drop(columns=["monsoon", "dow"], errors="ignore")
    d = add_reference(d, cal, ["dispatch_date"], [c for c in cal if c != "dispatch_date"])
    road = data.read("road_conditions.csv").rename(columns={"date": "dispatch_date"})
    d = add_reference(d, road, ["district", "dispatch_date"], ["disruption_index"])
    d = add_reference(d, data.read("service_allowance.csv"), ["brand", "dock_type"], ["service_allowance_min"])
    for col in ["planned_arrival_time", "planned_depart_time", "window_open_time", "window_close_time"]:
        d[col + "_minutes"] = minutes(d[col])
    d["hour"] = (d.planned_depart_time_minutes // 60).astype(int)
    d = add_reference(d, data.read("traffic_speed.csv"), ["district", "hour", "monsoon"], ["speed_index"])
    d["planned_slack_min"] = d.window_close_time_minutes - d.planned_arrival_time_minutes
    d["planned_wait_min"] = (d.window_open_time_minutes - d.planned_arrival_time_minutes).clip(lower=0)
    d["window_width_min"] = d.window_close_time_minutes - d.window_open_time_minutes
    d["traffic_adjusted_travel"] = (d.planned_travel_duration_min * 100 / d.speed_index.clip(lower=1)
                                    * 100 / d.disruption_index.clip(lower=1))
    d["extra_travel_estimate"] = d.traffic_adjusted_travel - d.planned_travel_duration_min
    dt = pd.to_datetime(d.dispatch_date)
    d["order_to_dispatch_days"] = (dt - pd.to_datetime(d.order_date)).dt.days
    d["month"] = dt.dt.month
    d["day_of_month"] = dt.dt.day
    d["annual_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    d["annual_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    d["weight_per_unit"] = d.order_weight_kg / d.order_units.clip(lower=1)
    d["volume_per_unit"] = d.order_volume_m3 / d.order_units.clip(lower=1)
    d["density"] = d.order_weight_kg / d.order_volume_m3.clip(lower=0.001)
    for c in ["order_units", "order_weight_kg", "order_volume_m3"]:
        d["log_" + c] = np.log1p(d[c])
    d["weight_fraction"] = d.order_weight_kg / d.weight_cap_kg
    d["volume_fraction"] = d.order_volume_m3 / d.volume_cap_m3
    d = d.sort_values(["route_id", "seq_in_route"], kind="stable")
    g = d.groupby("route_id", sort=False)
    d["route_stops"] = g.delivery_id.transform("size")
    d["route_start_min"] = g.planned_depart_time_minutes.transform("min")
    d["planned_elapsed_min"] = d.planned_arrival_time_minutes - d.route_start_min
    for c in ["order_units", "order_weight_kg", "order_volume_m3", "service_allowance_min",
              "planned_travel_duration_min", "traffic_adjusted_travel", "extra_travel_estimate", "planned_wait_min"]:
        d["prior_" + c] = g[c].cumsum() - d[c]
    for c in ["order_weight_kg", "order_volume_m3"]:
        d["route_" + c] = g[c].transform("sum")
    d["accumulated_traffic_slack"] = d.planned_slack_min - d.prior_extra_travel_estimate - d.extra_travel_estimate
    d["brand_dock"] = d.brand + ":" + d.dock_type
    d["district_hour"] = d.district + ":" + d.hour.astype(str)
    d["brand_temp"] = d.brand + ":" + d.temp_requirement
    keep_strings = {"outlet_id", "brand", "district", "depot", "temp_requirement", "vehicle_type", "vehicle_temp",
                    "dock_type", "parking_constraint", "road_class", "festival", "brand_dock", "district_hour", "brand_temp"}
    exclude = {"_position", "iso_year", "iso_week"}
    cols = [c for c in d if c in keep_strings or (pd.api.types.is_numeric_dtype(d[c]) and c not in exclude)]
    d = d.sort_values("_position")[cols].reset_index(drop=True)
    for c in d.select_dtypes(include=["object", "string"]):
        d[c] = d[c].fillna("none").astype(str)
    return d.replace([np.inf, -np.inf], np.nan)


def metrics(y, pred, classification=False):
    y, pred = np.asarray(y), np.asarray(pred)
    require(len(y) == len(pred) and np.isfinite(pred).all(), "Bad predictions")
    if classification:
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        return {"log_loss": float(log_loss(y, pred, labels=[0, 1])),
                "brier": float(brier_score_loss(y, pred)),
                "roc_auc": float(roc_auc_score(y, pred)) if len(np.unique(y)) > 1 else None,
                "pr_auc": float(average_precision_score(y, pred)),
                "accuracy_at_0.5": float(np.mean((pred >= 0.5) == y))}
    err = np.abs(y - pred)
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred)),
            "wape": float(err.sum() / max(np.abs(y).sum(), 1e-9)),
            "smape": float(np.mean(2 * err / np.maximum(np.abs(y) + np.abs(pred), 1e-6)))}


def preprocess(x, linear=False):
    cat = list(x.select_dtypes(include=["object", "string", "category"]).columns)
    num = [c for c in x if c not in cat]
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]) if linear else SimpleImputer(strategy="median")
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if linear else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer([("num", numeric, num), ("cat", categorical, cat)], sparse_threshold=0)


class FittedModel:
    def __init__(self, name, estimator, prep, columns, kind, rounds=None):
        self.name, self.estimator, self.prep, self.columns, self.kind, self.rounds = name, estimator, prep, list(columns), kind, rounds

    def predict(self, x):
        x = x[self.columns]
        if self.prep is not None:
            x = self.prep.transform(x)
        if self.kind == "classification":
            return np.clip(self.estimator.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
        return np.maximum(0, self.estimator.predict(x))


def fit_model(spec, x, y, kind, jobs, seed=None, validation=None, fixed_rounds=None):
    """Explicit, documented model configurations; no automated end-to-end modeller."""
    seed = SEED if seed is None else seed
    family = spec["family"]
    p = {k: v for k, v in spec.items() if k not in {"name", "family"}}
    is_class = kind == "classification"
    configured_rounds = p.pop("iterations", 500)
    rounds = fixed_rounds or configured_rounds
    prep = None
    xfit, xval = x, validation[0] if validation else None
    if family not in {"catboost"}:
        prep = preprocess(x, linear=family in {"ridge", "logistic"})
        xfit = prep.fit_transform(x)
        xval = prep.transform(xval) if xval is not None else None
    if family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if is_class else CatBoostRegressor
        estimator = cls(iterations=rounds, loss_function="Logloss" if is_class else "RMSE",
                        random_seed=seed, thread_count=jobs, verbose=False, allow_writing_files=False,
                        cat_features=list(x.select_dtypes(include=["object", "string"]).columns), **p)
        kw = {"eval_set": (xval, validation[1]), "early_stopping_rounds": 100, "use_best_model": True} if validation else {}
        estimator.fit(xfit, y, **kw)
        rounds = int(estimator.tree_count_)
    elif family == "lightgbm":
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if is_class else lgb.LGBMRegressor
        estimator = cls(n_estimators=rounds, random_state=seed, n_jobs=jobs, verbosity=-1,
                        deterministic=True, force_col_wise=True, **p)
        kw = {"eval_set": [(xval, validation[1])], "callbacks": [lgb.early_stopping(80, verbose=False)]} if validation else {}
        estimator.fit(xfit, y, **kw)
        rounds = int(estimator.best_iteration_ or rounds)
    elif family == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if is_class else XGBRegressor
        estimator = cls(n_estimators=rounds, objective="binary:logistic" if is_class else "reg:squarederror",
                        tree_method="hist", n_jobs=jobs, random_state=seed,
                        **({"early_stopping_rounds": 80} if validation else {}), **p)
        estimator.fit(xfit, y, **({"eval_set": [(xval, validation[1])], "verbose": False} if validation else {}))
        if validation:
            rounds = int(estimator.best_iteration + 1)
    elif family == "hist":
        cls = HistGradientBoostingClassifier if is_class else HistGradientBoostingRegressor
        estimator = cls(max_iter=rounds, early_stopping=False, random_state=seed, **p).fit(xfit, y)
    elif family in {"extra", "forest"}:
        cls = (ExtraTreesClassifier if is_class else ExtraTreesRegressor) if family == "extra" else (RandomForestClassifier if is_class else RandomForestRegressor)
        estimator = cls(n_estimators=rounds, random_state=seed, n_jobs=jobs, **p).fit(xfit, y)
    elif family == "ridge":
        estimator = Ridge(**p).fit(xfit, y)
    elif family == "logistic":
        estimator = LogisticRegression(max_iter=1500, **p).fit(xfit, y)
    else:
        raise ValueError(f"Unknown model family {family}")
    return FittedModel(spec["name"], estimator, prep, x.columns, kind, rounds)


def model_specs(preset, classification=False):
    n = {"quick": 350, "balanced": 1100, "thorough": 2200}[preset]
    specs = [
        {"name": "catboost_depth6", "family": "catboost", "depth": 6, "learning_rate": .045, "l2_leaf_reg": 5, "iterations": n},
        {"name": "lightgbm_leaves31", "family": "lightgbm", "num_leaves": 31, "learning_rate": .04, "reg_lambda": 5, "min_child_samples": 45, "iterations": n},
        {"name": "hist_leaves31", "family": "hist", "max_leaf_nodes": 31, "learning_rate": .06, "l2_regularization": 8, "min_samples_leaf": 45, "iterations": min(n, 400)},
        {"name": "linear_baseline", "family": "logistic" if classification else "ridge", **({"C": .5} if classification else {"alpha": 20})},
    ]
    if preset != "quick":
        specs += [
            {"name": "catboost_depth8", "family": "catboost", "depth": 8, "learning_rate": .04, "l2_leaf_reg": 8, "iterations": n},
            {"name": "lightgbm_leaves63", "family": "lightgbm", "num_leaves": 63, "learning_rate": .035, "reg_lambda": 15, "min_child_samples": 70, "iterations": n},
            {"name": "xgboost_depth5", "family": "xgboost", "max_depth": 5, "learning_rate": .04, "reg_lambda": 10, "subsample": .9, "colsample_bytree": .9, "iterations": n},
            {"name": "extra_trees", "family": "extra", "iterations": 180, "min_samples_leaf": 5, "max_features": .85},
        ]
    if preset == "thorough":
        specs += [
            {"name": "catboost_depth5", "family": "catboost", "depth": 5, "learning_rate": .03, "l2_leaf_reg": 3, "iterations": n},
            {"name": "catboost_depth7", "family": "catboost", "depth": 7, "learning_rate": .025, "l2_leaf_reg": 12, "iterations": n},
            {"name": "lightgbm_leaves15", "family": "lightgbm", "num_leaves": 15, "learning_rate": .025, "reg_lambda": 3, "min_child_samples": 30, "iterations": n},
            {"name": "xgboost_depth7", "family": "xgboost", "max_depth": 7, "learning_rate": .025, "reg_lambda": 15, "subsample": .85, "colsample_bytree": .85, "iterations": n},
            {"name": "random_forest", "family": "forest", "iterations": 180, "min_samples_leaf": 6, "max_features": .8, "max_samples": .8},
        ]
    # Optional libraries may be absent: record the skip, keep sklearn alternatives runnable.
    available = []
    for s in specs:
        if s["family"] in {"catboost", "lightgbm", "xgboost"} and importlib.util.find_spec(s["family"]) is None:
            LOG.warning("SKIPPED %s: install %s to compare it", s["name"], s["family"])
        else:
            available.append(s)
    return available


def blend_weights(y, predictions, classification=False):
    matrix = np.column_stack(predictions)
    def loss(w):
        p = matrix @ w
        return log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)) if classification else np.mean((y - p) ** 2)
    result = minimize(loss, np.ones(matrix.shape[1]) / matrix.shape[1], method="SLSQP",
                      bounds=[(0, 1)] * matrix.shape[1], constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                      options={"maxiter": 150, "ftol": 1e-10})
    if not result.success:
        best = min(range(matrix.shape[1]), key=lambda j: loss(np.eye(matrix.shape[1])[j]))
        return np.eye(matrix.shape[1])[best]
    w = np.maximum(result.x, 0)
    w[w < .01] = 0
    return w / w.sum()


class Ensemble:
    def __init__(self, models, weights, calibrator=None):
        self.models, self.weights, self.calibrator = models, np.asarray(weights), calibrator

    def predict(self, x):
        p = sum(w * m.predict(x) for w, m in zip(self.weights, self.models))
        if self.calibrator is not None:
            logits = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
            p = self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        return p


def run_task1(data, out, args):
    x, service, late, meta = construct_task1(data, True)
    test_x, _, _, test_meta = construct_task1(data, False)
    require(set(x.columns) == set(test_x.columns), "Train/test feature schema differs")
    # 6-week final holdout matches the task1 test span. Two prior 4-week folds tune.
    end = meta.dispatch_date.max().normalize() + pd.Timedelta(days=1)
    hold_start = end - pd.Timedelta(days=42)
    tuning_spans = [(hold_start - pd.Timedelta(days=56), hold_start - pd.Timedelta(days=28)),
                    (hold_start - pd.Timedelta(days=28), hold_start)]
    hold = (meta.dispatch_date >= hold_start).to_numpy()
    development = ~hold
    require(meta.loc[development, "dispatch_date"].max() < meta.loc[hold, "dispatch_date"].min(), "Invalid time split")
    require(set(meta.loc[development, "route_id"]).isdisjoint(meta.loc[hold, "route_id"]), "Route crosses split")
    report = {"labeled_orders": len(x), "not_run_excluded": len(data.read("deliveries_train.csv")) - len(x),
              "features": len(x.columns), "label_service": "leave_outlet - max(arrival, window_open)",
              "label_late": "arrival > window_close", "late_prevalence": float(late.mean()),
              "tuning_windows": [[str(a.date()), str((b - pd.Timedelta(days=1)).date())] for a, b in tuning_spans],
              "holdout": [str(hold_start.date()), str((end - pd.Timedelta(days=1)).date())],
              "selection_note": "Tune candidates, weights and calibration before the untouched holdout; refit on all labels afterwards."}
    final_predictions = {}
    boards = []
    for target, y, classification in [("service", service, False), ("lateness", late, True)]:
        kind = "classification" if classification else "regression"
        primary = "log_loss" if classification else "rmse"
        specs = model_specs(args.preset, classification)
        predictions, rounds_by_name, valid_specs = {}, {}, {}
        tune_truth, tune_indices = [], []
        for a, b in tuning_spans:
            idx = np.flatnonzero(((meta.dispatch_date >= a) & (meta.dispatch_date < b)).to_numpy())
            tune_indices.extend(idx)
            tune_truth.extend(y[idx])
        tune_truth = np.array(tune_truth)
        for spec in specs:
            started = time.time()
            preds, used_rounds = [], []
            for a, b in tuning_spans:
                train = (meta.dispatch_date < a).to_numpy()
                val = ((meta.dispatch_date >= a) & (meta.dispatch_date < b)).to_numpy()
                require(train.sum() > 1000 and val.sum() > 100, "Insufficient chronological training data")
                model = fit_model(spec, x.loc[train], y[train], kind, args.jobs,
                                  validation=(x.loc[val], y[val]))
                preds.append(model.predict(x.loc[val]))
                used_rounds.append(model.rounds)
            p = np.concatenate(preds)
            score = metrics(tune_truth, p, classification)
            predictions[spec["name"]] = p
            rounds_by_name[spec["name"]] = int(np.median(used_rounds))
            valid_specs[spec["name"]] = spec
            boards.append({"target": target, "model": spec["name"], "stage": "tuning", "seconds": round(time.time() - started, 2), **score})
            pd.DataFrame(boards).to_csv(out / "reports/task1_leaderboard.csv", index=False)
            LOG.info("%s / %s: %s=%.5f (%.1fs)", target, spec["name"], primary, score[primary], time.time() - started)
        ranked = sorted(predictions, key=lambda n: metrics(tune_truth, predictions[n], classification)[primary])[:4]
        weights = blend_weights(tune_truth, [predictions[n] for n in ranked], classification)
        selected = [(n, float(w)) for n, w in zip(ranked, weights) if w > 0]
        blended = sum(w * predictions[n] for n, w in selected)
        calibrator = None
        calibration_report = {"method": "none"}
        if classification:
            # Assess sigmoid calibration on the second tuning fold, fit on the first.
            first_len = sum((meta.dispatch_date >= tuning_spans[0][0]) & (meta.dispatch_date < tuning_spans[0][1]))
            logits = np.log(blended / (1 - blended)).reshape(-1, 1)
            trial = LogisticRegression(C=10).fit(logits[:first_len], tune_truth[:first_len])
            cp = trial.predict_proba(logits[first_len:])[:, 1]
            before = log_loss(tune_truth[first_len:], blended[first_len:])
            after = log_loss(tune_truth[first_len:], cp)
            calibration_report.update({"second_tuning_fold_before": float(before), "second_tuning_fold_after": float(after),
                                       "caveat": "Ensemble weights used both tuning folds; only final holdout is independent."})
            if after < before:
                calibrator = LogisticRegression(C=10).fit(logits, tune_truth)
                calibration_report["method"] = "sigmoid_on_oof_probabilities"
        frozen_models = []
        for name, _ in selected:
            frozen_models.append(fit_model(valid_specs[name], x.loc[development], y[development], kind,
                                          args.jobs, fixed_rounds=rounds_by_name[name]))
        frozen = Ensemble(frozen_models, [w for _, w in selected], calibrator)
        hp = frozen.predict(x.loc[hold])
        hold_score = metrics(y[hold], hp, classification)
        LOG.info("UNTOUCHED HOLDOUT %s: %s", target, hold_score)
        boards.append({"target": target, "model": "frozen_ensemble", "stage": "holdout", **hold_score})
        pd.DataFrame(boards).to_csv(out / "reports/task1_leaderboard.csv", index=False)
        hold_rows = meta.loc[hold].copy()
        hold_rows["actual"] = y[hold]
        hold_rows["prediction"] = hp
        hold_rows.to_csv(out / f"reports/task1_{target}_holdout.csv", index=False)
        by_brand = {b: metrics(y[hold][(meta.loc[hold, "brand"] == b).to_numpy()], hp[(meta.loc[hold, "brand"] == b).to_numpy()], classification)
                    for b in sorted(meta.brand.unique())}
        final_models = [fit_model(valid_specs[name], x, y, kind, args.jobs, fixed_rounds=rounds_by_name[name]) for name, _ in selected]
        ensemble = Ensemble(final_models, [w for _, w in selected], calibrator)
        joblib.dump(ensemble, out / f"models/task1_{target}.joblib", compress=3)
        final_predictions[target] = ensemble.predict(test_x)
        report[target] = {"selection": selected, "rounds": {n: rounds_by_name[n] for n, _ in selected},
                          "holdout_metrics": hold_score, "holdout_by_brand": by_brand, "calibration": calibration_report}
        # Average normalized feature importances, when supported.
        importances = []
        for name, w in selected:
            model = final_models[[n for n, _ in selected].index(name)]
            if hasattr(model.estimator, "feature_importances_"):
                vals = np.asarray(model.estimator.feature_importances_, float)
                cols = model.columns if model.prep is None else [str(c).split("__", 1)[-1] for c in model.prep.get_feature_names_out()]
                if len(cols) == len(vals):
                    importances.extend({"model": name, "feature": c, "importance": float(v), "ensemble_weight": w} for c, v in zip(cols, vals))
        if importances:
            pd.DataFrame(importances).to_csv(out / f"reports/task1_{target}_feature_importance.csv", index=False)
    pred = pd.DataFrame({"delivery_id": test_meta.delivery_id, "pred_service_min": final_predictions["service"],
                         "pred_late_prob": np.clip(final_predictions["lateness"], 0, 1)})
    template = data.read("submission_task1.csv")
    sub = template[["delivery_id"]].merge(pred, on="delivery_id", validate="one_to_one", how="left")
    require(sub.delivery_id.equals(template.delivery_id), "Task1 template row order changed")
    require(sub.notna().all().all(), "Task1 predictions missing")
    sub.to_csv(out / "submissions/submission_task1.csv", index=False, float_format="%.8f")
    write_json(out / "reports/task1_metrics.json", report)
    return report


def demand_panel(data):
    """Zero-filled calendar panel. Count requested orders once, including not_run."""
    a, b = data.read("deliveries_train.csv"), data.read("task1_test_inputs.csv")
    orders = pd.concat([a, b], ignore_index=True)
    require(not orders.delivery_id.duplicated().any(), "Task2A demand orders overlap or duplicate")
    require((orders.order_volume_m3 >= 0).all(), "Negative order volume")
    cal = data.read("calendar.csv")
    cal["date"] = pd.to_datetime(cal.date)
    require(not cal.date.duplicated().any(), "Duplicate calendar date")
    orders["date"] = pd.to_datetime(orders.order_date)  # NOT dispatch_date.
    orders["total"] = orders.order_volume_m3
    orders["chilled"] = np.where(orders.temp_requirement.eq("chilled"), orders.total, 0)
    require(orders.loc[orders.brand.ne("Fresh"), "chilled"].eq(0).all(), "Unexpected non-Fresh chilled demand")
    require(orders.date.isin(cal.date).all(), "Order date absent from calendar")
    orders = orders.merge(cal[["date", "is_operating"]], on="date", validate="many_to_one")
    require(orders.is_operating.eq(1).all(), "Demand exists on a non-operating day; inspect before filling zeros")
    agg = orders.groupby(["date", "depot", "brand"])[["total", "chilled"]].sum().reset_index()
    groups = data.read("outlets.csv")[["depot", "brand"]].drop_duplicates()
    panel = cal.merge(groups, how="cross").merge(agg, on=["date", "depot", "brand"], how="left", validate="one_to_one")
    panel[["total", "chilled"]] = panel[["total", "chilled"]].fillna(0)
    panel["week_start"] = panel.date - pd.to_timedelta(panel.dow, unit="D")
    last_observed, first_observed = orders.date.max(), orders.date.min()
    # Only complete ISO weeks can enter weekly training/evaluation; Sunday's zero
    # is known from the calendar even when the last order is on Saturday.
    weeks = cal.assign(week_start=cal.date - pd.to_timedelta(cal.dow, unit="D")).groupby("week_start").agg(
        dates=("date", "size"), last_date=("date", "max"))
    op_last = cal.loc[cal.is_operating.eq(1)].assign(
        week_start=lambda z: z.date - pd.to_timedelta(z.dow, unit="D")).groupby("week_start").date.max()
    complete = weeks.index[(weeks.dates == 7) & (weeks.index >= first_observed)
                           & (op_last.reindex(weeks.index) <= last_observed)]
    history = panel[panel.week_start.isin(complete)].copy().reset_index(drop=True)
    require(abs(history.total.sum() - orders.total.sum()) < 1e-6, "Partial history week detected: do not silently discard orders")
    future = data.read("task2a_test_inputs.csv")
    require(not future.duplicated(["depot", "brand", "iso_year", "iso_week"]).any(), "Duplicate forecast key")
    week_keys = panel[["iso_year", "iso_week", "week_start"]].drop_duplicates()
    future = future.merge(week_keys, on=["iso_year", "iso_week"], how="left", validate="many_to_one")
    require(future.week_start.notna().all(), "Forecast week missing from calendar")
    require(history.week_start.max() < future.week_start.min(), "Forecast overlaps observed demand")
    future_days = panel.merge(future[["depot", "brand", "iso_year", "iso_week"]],
                              on=["depot", "brand", "iso_year", "iso_week"], validate="many_to_one")
    require(future_days.groupby(["depot", "brand", "iso_year", "iso_week"]).size().eq(7).all(), "Incomplete future week")
    return history, future_days, future, {"orders_counted": len(orders),
        "not_run_orders_counted": int(orders.dispatch_status.eq("not_run").sum()),
        "deferred_orders_counted": int(orders.dispatch_status.eq("deferred").sum()),
        "historical_weeks": int(history.week_start.nunique()), "last_order_date": str(last_observed.date()),
        "demand_date": "order_date", "weekly_keys": "calendar.iso_year + calendar.iso_week",
        "task1_inputs_used_for_demand": len(b)}


def calendar_features(panel):
    """Only known future calendar context; no future order or dispatch information."""
    x = panel[["depot", "brand", "dow", "is_payday", "festival_ramp", "is_holiday", "monsoon", "is_operating"]].copy()
    dt = pd.to_datetime(panel.date)
    x["series"] = panel.depot + ":" + panel.brand
    x["weekday"] = panel.dow.astype(str)
    x["month"] = dt.dt.month.astype(str)
    x["month_day"] = dt.dt.day
    x["festival_ramp_squared"] = panel.festival_ramp ** 2
    x["festival_near"] = (panel.festival_ramp > 0).astype(int)
    for harmonic in (1, 2, 3):
        x[f"sin_year_{harmonic}"] = np.sin(2 * np.pi * harmonic * dt.dt.dayofyear / 365.25)
        x[f"cos_year_{harmonic}"] = np.cos(2 * np.pi * harmonic * dt.dt.dayofyear / 365.25)
    for day in range(6):
        indicator = (panel.dow == day).astype(int)
        x[f"dow_{day}"] = indicator
        x[f"festival_on_{day}"] = indicator * panel.festival_ramp
        x[f"payday_on_{day}"] = indicator * panel.is_payday
    return x


def weekly_context(panel):
    aggregations = {"total": "sum", "chilled": "sum", "is_operating": "sum", "is_payday": "sum",
                    "festival_ramp": "sum", "is_holiday": "sum", "monsoon": "mean"}
    w = panel.groupby(["depot", "brand", "iso_year", "iso_week", "week_start"], as_index=False).agg(aggregations)
    w["week_sin"] = np.sin(2 * np.pi * w.iso_week / 52.1775)
    w["week_cos"] = np.cos(2 * np.pi * w.iso_week / 52.1775)
    w["series"] = w.depot + ":" + w.brand
    return w


class DemandModel:
    def __init__(self, name, target, jobs=4, preset="balanced"):
        self.name, self.target, self.jobs, self.preset = name, target, jobs, preset

    def fit(self, panel):
        target = self.target
        p = panel[panel.brand.eq("Fresh")].copy() if target == "chilled" else panel.copy()
        self.history = weekly_context(p)
        self.cutoff = p.week_start.max()
        self.scales = p.loc[p.is_operating.eq(1)].groupby(["depot", "brand"])[target].mean().clip(lower=.01).to_dict()
        if self.name.startswith("mean_") or self.name == "seasonal_52":
            return self
        if self.name.startswith("ridge_"):
            alpha = float(self.name.split("_")[-1])
            self.models = {}
            for key, g in p[p.is_operating.eq(1)].groupby(["depot", "brand"]):
                x = calendar_features(g)
                self.models[key] = fit_model({"name": self.name, "family": "ridge", "alpha": alpha}, x, g[target].to_numpy(),
                                              "regression", self.jobs)
            return self
        if self.name == "weekly_catboost":
            w = self.history
            features = ["depot", "brand", "series", "iso_week", "is_operating", "is_payday", "festival_ramp",
                        "is_holiday", "monsoon", "week_sin", "week_cos"]
            self.features = features
            scale = np.array([self.scales[(r.depot, r.brand)] * 6 for r in w.itertuples()])
            spec = {"name": self.name, "family": "catboost", "depth": 3, "learning_rate": .035, "l2_leaf_reg": 12, "iterations": 450}
            self.model = fit_model(spec, w[features], w[target].to_numpy() / scale, "regression", self.jobs)
            return self
        p = p[p.is_operating.eq(1)]
        scale = np.array([self.scales[(r.depot, r.brand)] for r in p.itertuples()])
        if self.name == "daily_catboost":
            spec = {"name": self.name, "family": "catboost", "depth": 5, "learning_rate": .035, "l2_leaf_reg": 12,
                    "iterations": 650 if self.preset != "quick" else 250}
        elif self.name == "daily_lightgbm":
            spec = {"name": self.name, "family": "lightgbm", "num_leaves": 15, "learning_rate": .035, "reg_lambda": 15,
                    "min_child_samples": 45, "iterations": 450 if self.preset != "quick" else 200}
        else:
            raise ValueError(self.name)
        self.model = fit_model(spec, calendar_features(p), p[target].to_numpy() / scale, "regression", self.jobs)
        return self

    def predict(self, panel):
        p = panel.copy().reset_index(drop=True)
        w = weekly_context(p)
        w["prediction"] = 0.0
        valid = w.brand.eq("Fresh") if self.target == "chilled" else pd.Series(True, index=w.index)
        if self.name.startswith("mean_") or self.name == "seasonal_52":
            for i, r in w[valid].iterrows():
                h = self.history[(self.history.depot == r.depot) & (self.history.brand == r.brand)].sort_values("week_start")
                if self.name == "seasonal_52":
                    lag = h[h.week_start == r.week_start - pd.Timedelta(weeks=52)]
                    value = float(lag[self.target].iloc[0]) if len(lag) else float(h[self.target].tail(13).mean())
                else:
                    value = float(h[self.target].tail(int(self.name.split("_")[1])).mean())
                w.loc[i, "prediction"] = value
        elif self.name == "weekly_catboost":
            scale = np.array([self.scales[(r.depot, r.brand)] * 6 for r in w[valid].itertuples()])
            w.loc[valid, "prediction"] = self.model.predict(w.loc[valid, self.features]) * scale
        else:
            p["prediction"] = 0.0
            mask = p.is_operating.eq(1)
            if self.target == "chilled":
                mask &= p.brand.eq("Fresh")
            if self.name.startswith("ridge_"):
                for key, model in self.models.items():
                    m = mask & p.depot.eq(key[0]) & p.brand.eq(key[1])
                    if m.any():
                        p.loc[m, "prediction"] = model.predict(calendar_features(p.loc[m]))
            else:
                scale = np.array([self.scales[(r.depot, r.brand)] for r in p[mask].itertuples()])
                p.loc[mask, "prediction"] = self.model.predict(calendar_features(p[mask])) * scale
            sums = p.groupby(["depot", "brand", "iso_year", "iso_week"], as_index=False).prediction.sum()
            w = w.drop(columns="prediction").merge(sums, on=["depot", "brand", "iso_year", "iso_week"], validate="one_to_one")
        return w


class DemandEnsemble:
    def __init__(self, models, selection):
        self.models, self.selection = models, selection

    def predict(self, panel):
        candidates = {name: model.predict(panel).set_index(["depot", "brand", "iso_year", "iso_week"])
                      for name, model in self.models.items()}
        result = next(iter(candidates.values())).copy()
        result["prediction"] = 0.0
        for key, terms in self.selection.items():
            mask = (result.index.get_level_values("depot") == key[0]) & (result.index.get_level_values("brand") == key[1])
            for name, weight in terms:
                result.loc[mask, "prediction"] += weight * candidates[name].reindex(result.index).loc[mask, "prediction"]
        return result.reset_index()


def run_task2a(data, out, args):
    history, future_days, future, report = demand_panel(data)
    weeks = sorted(history.week_start.unique())
    require(len(weeks) >= 70, "Need at least 70 complete weeks for rolling-origin evaluation")
    # Ten-week frozen-origin horizons, exactly like the competition test.
    hold_start = pd.Timestamp(weeks[-10])
    fold_starts = [pd.Timestamp(weeks[-30]), pd.Timestamp(weeks[-20])]
    report["tuning_origins"] = [str(d.date()) for d in fold_starts]
    report["untouched_holdout_start"] = str(hold_start.date())
    report["horizon_weeks"] = 10
    names = ["mean_4", "mean_13", "mean_26", "seasonal_52", "ridge_1", "ridge_30", "ridge_100"]
    if importlib.util.find_spec("catboost"):
        names.extend(["daily_catboost", "weekly_catboost"])
    if importlib.util.find_spec("lightgbm"):
        names.append("daily_lightgbm")
    boards, target_forecasts = [], {}
    hold_predictions = {}
    for target in ["total", "chilled"]:
        per_candidate = {}
        for name in names:
            fold_rows = []
            for start in fold_starts:
                tr = history[history.week_start < start]
                va = history[(history.week_start >= start) & (history.week_start < start + pd.Timedelta(weeks=10))]
                model = DemandModel(name, target, args.jobs, args.preset).fit(tr)
                pred = model.predict(va)
                pred["origin"] = start
                fold_rows.append(pred)
            pred = pd.concat(fold_rows, ignore_index=True)
            per_candidate[name] = pred
            active = pred.brand.eq("Fresh") if target == "chilled" else pd.Series(True, index=pred.index)
            score = metrics(pred.loc[active, target], pred.loc[active, "prediction"])
            boards.append({"target": target, "model": name, "stage": "tuning", **score})
            LOG.info("Demand %s / %s: WAPE %.4f, RMSE %.4f", target, name, score["wape"], score["rmse"])
        selection = {}
        groups = history[["depot", "brand"]].drop_duplicates().itertuples(index=False, name=None)
        for key in groups:
            if target == "chilled" and key[1] != "Fresh":
                continue
            losses = {}
            for name, p in per_candidate.items():
                g = p[p.depot.eq(key[0]) & p.brand.eq(key[1])]
                losses[name] = metrics(g[target], g.prediction)["rmse"]
                boards.append({"target": target, "model": name, "stage": "tuning_series", "depot": key[0], "brand": key[1], **metrics(g[target], g.prediction)})
            top = sorted(losses, key=losses.get)[:3]
            best_rows = per_candidate[top[0]]
            y = best_rows.loc[best_rows.depot.eq(key[0]) & best_rows.brand.eq(key[1]), target].to_numpy()
            predictions = [p.loc[p.depot.eq(key[0]) & p.brand.eq(key[1]), "prediction"].to_numpy() for p in [per_candidate[n] for n in top]]
            w = blend_weights(y, predictions)
            selection[key] = [(n, float(v)) for n, v in zip(top, w) if v > 0]
        chosen = sorted({n for terms in selection.values() for n, _ in terms})
        tr = history[history.week_start < hold_start]
        hold = history[history.week_start >= hold_start]
        frozen = DemandEnsemble({n: DemandModel(n, target, args.jobs, args.preset).fit(tr) for n in chosen}, selection)
        hp = frozen.predict(hold)
        hold_predictions[target] = hp
        final = DemandEnsemble({n: DemandModel(n, target, args.jobs, args.preset).fit(history) for n in chosen}, selection)
        joblib.dump(final, out / f"models/task2a_{target}.joblib", compress=3)
        target_forecasts[target] = final.predict(future_days)
        report[target] = {"selection": {":".join(k): v for k, v in selection.items()}}
    # Reconcile both holdout and future with the exact same physical constraints.
    key = ["depot", "brand", "iso_year", "iso_week"]
    for result_set, label in [(hold_predictions, "holdout"), (target_forecasts, "forecast")]:
        combined = result_set["total"][key + ["total", "chilled", "prediction"]].rename(columns={"prediction": "pred_total_volume_m3"})
        combined = combined.merge(result_set["chilled"][key + ["prediction"]].rename(columns={"prediction": "pred_chilled_volume_m3"}), on=key, validate="one_to_one")
        combined["pred_total_volume_m3"] = combined.pred_total_volume_m3.clip(lower=0)
        combined["pred_chilled_volume_m3"] = np.minimum(combined.pred_chilled_volume_m3.clip(lower=0), combined.pred_total_volume_m3)
        combined.loc[combined.brand.ne("Fresh"), "pred_chilled_volume_m3"] = 0
        if label == "holdout":
            for target, col in [("total", "pred_total_volume_m3"), ("chilled", "pred_chilled_volume_m3")]:
                m = combined.brand.eq("Fresh") if target == "chilled" else pd.Series(True, index=combined.index)
                score = metrics(combined.loc[m, target], combined.loc[m, col])
                report[target]["holdout_metrics"] = score
                report[target]["holdout_by_series"] = {f"{k[0]}:{k[1]}": metrics(g[target], g[col]) for k, g in combined[m].groupby(["depot", "brand"])}
                boards.append({"target": target, "model": "frozen_ensemble", "stage": "holdout", **score})
                LOG.info("UNTOUCHED HOLDOUT demand %s: %s", target, score)
            combined.to_csv(out / "reports/task2a_holdout.csv", index=False)
        else:
            pred = future.merge(combined[key + ["pred_total_volume_m3", "pred_chilled_volume_m3"]], on=key, how="left", validate="one_to_one")
            template = data.read("submission_task2a.csv")
            sub = template[["row_id"]].merge(pred[["row_id", "pred_total_volume_m3", "pred_chilled_volume_m3"]], on="row_id", how="left", validate="one_to_one")
            require(sub.row_id.equals(template.row_id) and sub.notna().all().all(), "Task2A rows missing or reordered")
            sub.to_csv(out / "submissions/submission_task2a.csv", index=False, float_format="%.8f")
    pd.DataFrame(boards).to_csv(out / "reports/task2a_leaderboard.csv", index=False)
    write_json(out / "reports/task2a_metrics.json", report)
    return report


DEFAULT_PRIORITY = {"base": 100, "Fresh": 300, "Style": 0, "Tech": 0,
                    "chilled_bonus": 150, "deferred_yesterday_bonus": 200,
                    "per_day_unserved": 15, "days_cap": 7}


def priority_points(order, policy):
    return (policy["base"] + policy[order.brand]
            + policy["chilled_bonus"] * int(order.temp_requirement == "chilled")
            + policy["deferred_yesterday_bonus"] * int(order.deferred_yesterday)
            + policy["per_day_unserved"] * min(int(order.days_since_last_served), policy["days_cap"]))


def allocation_references(data):
    orders = data.read("task2b_peak_day_scenarios.csv")
    fleet = data.read("task2b_peak_day_fleet.csv")
    require(not orders.duplicated(["scenario", "order_ref"]).any(), "Duplicate scenario order")
    require(not fleet.duplicated(["scenario", "vehicle_id"]).any(), "Duplicate fleet status")
    vehicles = data.read("vehicles.csv")
    available = fleet[fleet.status.eq("available")].merge(vehicles, on="vehicle_id", validate="many_to_one")
    travel = data.read("district_travel.csv").set_index("district").to_dict("index")
    allowance = {(r.brand, r.dock_type): float(r.service_allowance_min) for r in data.read("service_allowance.csv").itertuples()}
    return orders, available, vehicles, travel, allowance


def compatible(order, vehicle, travel, allowance):
    if order.scenario != vehicle.scenario or order.depot != vehicle.depot:
        return False
    if order.temp_requirement == "chilled" and vehicle.temp != "reefer":
        return False
    if order.parking_constraint == "van_only" and vehicle.type != "van":
        return False
    if order.order_volume_m3 > vehicle.volume_cap_m3 + 1e-9 or order.order_weight_kg > vehicle.weight_cap_kg + 1e-9:
        return False
    single_time = travel[order.district]["depot_to_district_freeflow_min"] + allowance[(order.brand, order.dock_type)]
    return single_time <= (270 if order.brand == "Fresh" else 480)


def solve_allocation(data, seconds, jobs, policy, objective="priority"):
    """Binary assignment model for the published Task2B rules, using CP-SAT."""
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError("Fleet allocation needs OR-Tools. Run: python -m pip install ortools") from exc
    orders, available, vehicles, travel, allowance = allocation_references(data)
    o, v = list(orders.itertuples(index=False)), list(available.itertuples(index=False))
    model = cp_model.CpModel()
    assignments, groups = {}, {}
    points = [priority_points(row, policy) for row in o]
    compatible_counts = []
    for i, order in enumerate(o):
        eligible = [j for j, vehicle in enumerate(v) if compatible(order, vehicle, travel, allowance)]
        compatible_counts.append(len(eligible))
        for j in eligible:
            for t in (1, 2):
                assignments[i, j, t] = model.new_bool_var(f"order_{i}_vehicle_{j}_trip_{t}")
        model.add(sum(assignments[i, j, t] for j in eligible for t in (1, 2)) <= 1)
    # Every trip contains one brand and one district, with exact integer capacities.
    for j, vehicle in enumerate(v):
        time_fresh, time_day = [], []
        trip_active = []
        for t in (1, 2):
            by_group = {}
            for (i, jj, tt), x in assignments.items():
                if (jj, tt) == (j, t):
                    by_group.setdefault((o[i].brand, o[i].district), []).append((i, x))
            z_variables = []
            for (brand, district), pairs in by_group.items():
                z = model.new_bool_var(f"group_{j}_{t}_{brand}_{district}")
                groups[j, t, brand, district] = z
                z_variables.append(z)
                for _, x in pairs:
                    model.add(x <= z)
                model.add(z <= sum(x for _, x in pairs))
                d = travel[district]
                # duration = (outbound - interstop)*active + sum(interstop + service)*assigned.
                # Multiply minutes by 100 to preserve any two-decimal planning allowances.
                expr = int(round(100 * (d["depot_to_district_freeflow_min"] - d["inter_stop_freeflow_min"]))) * z
                expr += sum(int(round(100 * (d["inter_stop_freeflow_min"] + allowance[(brand, o[i].dock_type)]))) * x for i, x in pairs)
                (time_fresh if brand == "Fresh" else time_day).append(expr)
            model.add(sum(z_variables) <= 1)
            trip_active.append(sum(z_variables))
            xs = [(i, x) for (i, jj, tt), x in assignments.items() if (jj, tt) == (j, t)]
            model.add(sum(int(round(o[i].order_volume_m3 * 1000)) * x for i, x in xs) <= int(round(vehicle.volume_cap_m3 * 1000)))
            model.add(sum(int(round(o[i].order_weight_kg * 10)) * x for i, x in xs) <= int(round(vehicle.weight_cap_kg * 10)))
        model.add(sum(time_fresh) <= 27000)
        model.add(sum(time_day) <= 48000)
        # Trip numbers are arbitrary; remove symmetry and leave no trip 2 without trip 1.
        model.add(trip_active[0] >= trip_active[1])
    # Primary weighted service objective, secondary number served, then small volume.
    # Multiplier is larger than every possible secondary improvement together.
    volume_bound = sum(int(round(r.order_volume_m3 * 1000)) for r in o) + 1
    secondary_bound = len(o) * volume_bound + volume_bound
    if objective == "priority":
        score = sum((points[i] * secondary_bound + volume_bound + int(round(o[i].order_volume_m3 * 1000))) * x
                    for (i, _, _), x in assignments.items())
    else:
        score = sum(x for x in assignments.values())
    model.maximize(score)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = jobs
    solver.parameters.random_seed = SEED
    status = solver.solve(model)
    require(status in (cp_model.OPTIMAL, cp_model.FEASIBLE),
            f"Allocator returned {solver.status_name(status)} without a feasible incumbent; increase --allocator-seconds")
    template = data.read("submission_task2b.csv")
    result = template[["scenario", "order_ref", "outlet_id"]].copy()
    result["decision"], result["vehicle_id"], result["trip_id"] = "deferred", "", pd.Series(pd.NA, index=result.index, dtype="Int64")
    locations = {(r.scenario, r.order_ref): i for i, r in enumerate(result.itertuples())}
    for (i, j, t), x in assignments.items():
        if solver.value(x):
            row = locations[o[i].scenario, o[i].order_ref]
            result.loc[row, ["decision", "vehicle_id", "trip_id"]] = ["served", v[j].vehicle_id, t]
    served = result.decision.eq("served")
    summary = {"objective": objective, "solver_status": solver.status_name(status),
               "objective_value": float(solver.objective_value), "objective_bound": float(solver.best_objective_bound),
               "relative_gap": float(abs(solver.best_objective_bound - solver.objective_value) / max(1, abs(solver.objective_value))),
               "seconds": float(solver.wall_time), "served": int(served.sum()), "deferred": int((~served).sum()),
               "priority_points_served": int(sum(points[i] for i, row in enumerate(o)
                                                  if result.loc[locations[row.scenario, row.order_ref], "decision"] == "served")),
               "individually_impossible_orders": [o[i].order_ref for i, c in enumerate(compatible_counts) if c == 0]}
    return result, summary


def validate_allocation(data, sub):
    """Independent validation, also checking template outlet IDs and blank deferrals."""
    orders, avail, vehicles, travel, allowance = allocation_references(data)
    template = data.read("submission_task2b.csv")
    for c in ["scenario", "order_ref", "outlet_id"]:
        require(sub[c].reset_index(drop=True).equals(template[c]), f"Task2B changed {c}")
    require(sub.decision.isin(["served", "deferred"]).all(), "Invalid allocation decision")
    deferred = sub.decision.eq("deferred")
    require(sub.loc[deferred, "vehicle_id"].fillna("").eq("").all() and sub.loc[deferred, "trip_id"].isna().all(), "Deferred assignments must be blank")
    m = orders.merge(sub.drop(columns="outlet_id"), on=["scenario", "order_ref"], validate="one_to_one")
    m = m[m.decision.eq("served")].copy()
    require(m.trip_id.notna().all() and m.trip_id.isin([1, 2]).all(), "Invalid trip ID")
    a = {(r.scenario, r.vehicle_id) for r in avail.itertuples()}
    veh = vehicles.set_index("vehicle_id")
    trips = []
    for (scenario, vid, tid), g in m.groupby(["scenario", "vehicle_id", "trip_id"]):
        require((scenario, vid) in a, f"Vehicle unavailable: {vid}")
        vehicle = veh.loc[vid]
        require(g.brand.nunique() == g.district.nunique() == g.depot.nunique() == 1, "Mixed trip")
        require(g.depot.iloc[0] == vehicle.depot, "Wrong depot")
        require(not g.temp_requirement.eq("chilled").any() or vehicle.temp == "reefer", "Chilled on ambient vehicle")
        require(not g.parking_constraint.eq("van_only").any() or vehicle.type == "van", "Truck at van-only outlet")
        volume, weight = float(g.order_volume_m3.sum()), float(g.order_weight_kg.sum())
        require(volume <= vehicle.volume_cap_m3 + 1e-6 and weight <= vehicle.weight_cap_kg + 1e-6, "Capacity exceeded")
        brand, district = g.brand.iloc[0], g.district.iloc[0]
        d = travel[district]
        outbound = d["depot_to_district_freeflow_min"]
        interstop = (len(g) - 1) * d["inter_stop_freeflow_min"]
        service = sum(allowance[(brand, r.dock_type)] for r in g.itertuples())
        trips.append({"scenario": scenario, "vehicle_id": vid, "trip_id": int(tid), "brand": brand, "district": district,
                      "orders": len(g), "volume_m3": volume, "volume_cap_m3": vehicle.volume_cap_m3,
                      "weight_kg": weight, "weight_cap_kg": vehicle.weight_cap_kg, "outbound_min": outbound,
                      "inter_stop_min": interstop, "service_min": service, "trip_minutes": outbound + interstop + service})
    trips = pd.DataFrame(trips)
    if len(trips):
        for _, g in trips.groupby(["scenario", "vehicle_id"]):
            require(len(g) <= 2, "More than two trips")
            require(g.loc[g.brand.eq("Fresh"), "trip_minutes"].sum() <= 270 + 1e-6, "Fresh budget exceeded")
            require(g.loc[g.brand.ne("Fresh"), "trip_minutes"].sum() <= 480 + 1e-6, "Daytime budget exceeded")
    return trips


def run_task2b(data, out, args):
    policy = DEFAULT_PRIORITY.copy()
    if args.priority_config:
        policy.update(json.loads(Path(args.priority_config).read_text()))
    require(all(isinstance(v, int) and v >= 0 for v in policy.values()), "Priority weights must be nonnegative integers")
    sub, summary = solve_allocation(data, args.allocator_seconds, args.jobs, policy)
    trips = validate_allocation(data, sub)
    path = out / "submissions/submission_task2b.csv"
    sub.to_csv(path, index=False)
    trips.to_csv(out / "reports/task2b_trip_audit.csv", index=False)
    # Compare with a different, explicit service policy; do not label this official scoring.
    count_sub, count_summary = solve_allocation(data, args.allocator_seconds, args.jobs, policy, "count")
    validate_allocation(data, count_sub)
    count_sub.to_csv(out / "reports/alternative_max_count_allocation.csv", index=False)
    summary["max_count_comparison"] = count_summary
    orders, available, vehicles, travel, allowance = allocation_references(data)
    audit = orders.merge(sub.drop(columns="outlet_id"), on=["scenario", "order_ref"], validate="one_to_one")
    audit["priority_points"] = [priority_points(r, policy) for r in audit.itertuples()]
    audit["eligible_vehicle_count"] = [sum(compatible(r, v, travel, allowance) for v in available.itertuples()) for r in audit.itertuples()]
    audit["reason"] = ["served under priority policy" if r.decision == "served" else
                       "individually infeasible: no available vehicle fits all single-order rules" if r.eligible_vehicle_count == 0 else
                       "deferred under priority policy while eligible vehicles share capacity/trips/time with higher-value combinations"
                       for r in audit.itertuples()]
    audit.to_csv(out / "reports/task2b_order_decisions.csv", index=False)
    summary["priority_policy"] = policy
    summary["served_volume_m3"] = float(audit.loc[audit.decision.eq("served"), "order_volume_m3"].sum())
    summary["deferred_volume_m3"] = float(audit.loc[audit.decision.eq("deferred"), "order_volume_m3"].sum())
    summary["by_brand"] = audit.groupby(["brand", "decision"]).agg(orders=("order_ref", "size"), volume_m3=("order_volume_m3", "sum")).reset_index().to_dict("records")
    checker_path = Path(args.checker).expanduser().resolve() if args.checker else None
    if checker_path and checker_path.is_file():
        spec = importlib.util.spec_from_file_location("organizer_checker", checker_path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        # Only adapt file discovery in memory; never edit organizer rules or source.
        checker._find = lambda name: str(data.paths[name])
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            code = checker.check(str(path))
        (out / "reports/official_checker.txt").write_text(capture.getvalue(), encoding="utf-8")
        require(code == 0, "Official checker rejected allocation: " + capture.getvalue())
        summary["official_checker"] = capture.getvalue().strip()
    else:
        summary["official_checker"] = "NOT RUN: supply --checker path/to/check_allocation.py. Independent validation passed."
    available_reefers = available[available.temp.eq("reefer")]
    chilled_orders = orders[orders.temp_requirement.eq("chilled")]
    vanchilled = chilled_orders[chilled_orders.parking_constraint.eq("van_only")]
    reasons = audit.loc[audit.decision.eq("deferred"), ["order_ref", "brand", "order_volume_m3", "priority_points", "reason"]]
    reason_lines = "\n".join(f"- {r.order_ref}: {r.brand}, {r.order_volume_m3:.3f} m3, {r.priority_points} points; {r.reason}." for r in reasons.itertuples())
    text = f"""# Peak-day prioritization policy

Serve Fresh early, protect the cold chain, and increase priority for repeat deferrals. The transparent benefit per whole order is {policy['base']} base points + the brand weight (Fresh {policy['Fresh']}, Style {policy['Style']}, Tech {policy['Tech']}) + {policy['chilled_bonus']} for chilled goods + {policy['deferred_yesterday_bonus']} if deferred yesterday + {policy['per_day_unserved']} per day since service, capped at {policy['days_cap']} days. These are our policy choices, not organizer scoring weights. Among equal priority totals, prefer more orders, then volume. Change `priority_config.json` to examine another policy.

There are {len(available)} available vehicles, including {len(available_reefers)} refrigerated vehicles ({len(available_reefers[available_reefers.type.eq('van')])} refrigerated van). Chilled demand is {chilled_orders.order_volume_m3.sum():.3f} m3, including {vanchilled.order_volume_m3.sum():.3f} m3 at van-only outlets. Twice the available refrigerated volume is {2 * available_reefers.volume_cap_m3.sum():.3f} m3: an optimistic upper bound before district separation, weight, whole-order packing and time. Fleet-wide volume is not interchangeable with refrigerated or van-access capacity.

The selected plan serves {summary['served']} of {len(orders)} orders, delivers {summary['served_volume_m3']:.3f} m3 and defers {summary['deferred']} orders / {summary['deferred_volume_m3']:.3f} m3. Solver status: {summary['solver_status']}; objective gap {summary['relative_gap']:.6%}. A separate maximum-order-count run serves {count_summary['served']} orders (status {count_summary['solver_status']}). The priority plan scores {summary['priority_points_served']} policy points versus {count_summary['priority_points_served']} for the count plan. Optimality, when proven, is only for the stated objective and published model.

Every trip has one brand and district, respects both capacities, depot, refrigeration and van restrictions, and each vehicle has at most two trips. Duration = outbound + (orders - 1) x inter-stop + sum(brand/dock allowances). Fresh trip durations sum to at most 270 minutes per vehicle; Style/Tech durations sum to at most 480 minutes separately. No return leg is added. The detailed calculations are in `task2b_trip_audit.csv`.

Individually infeasible orders: {', '.join(summary['individually_impossible_orders']) or 'none'}. Other deferrals are choices under shared resources, not proven individually unavoidable. Their order-level reasons and opportunity cost (volume and policy points) are in `task2b_order_decisions.csv`. This is the published Task2B planning model: it does not claim a stop-by-stop route, weekly fuel ledger, or exact arrival-window feasibility beyond the supplied daily budgets.
"""
    (out / "reports/task2b_policy.md").write_text(text, encoding="utf-8")
    (out / "reports/task2b_deferrals.md").write_text("# Deferred orders\n\n" + reason_lines + "\n", encoding="utf-8")
    write_json(out / "reports/task2b_summary.json", summary)
    LOG.info("ALLOCATION: %s, %s served, %s deferred; checker: %s", summary["solver_status"], summary["served"], summary["deferred"], summary["official_checker"])
    return summary


def validate_submissions(data, out):
    folder = out / "submissions"
    report = {}
    for task, key in [("task1", ["delivery_id"]), ("task2a", ["row_id"]), ("task2b", ["scenario", "order_ref", "outlet_id"])]:
        name = f"submission_{task}.csv"
        template, sub = data.read(name), pd.read_csv(folder / name)
        require(list(template.columns) == list(sub.columns), f"Wrong columns: {name}")
        require(len(template) == len(sub), f"Wrong row count: {name}")
        require(template[key].equals(sub[key]), f"Identifiers or row order changed: {name}")
        require(not sub.duplicated(key).any(), f"Duplicate submission key: {name}")
        if task != "task2b":
            values = sub.drop(columns=key).to_numpy(float)
            require(np.isfinite(values).all() and (values >= 0).all(), f"Invalid numeric predictions: {name}")
        if task == "task1":
            require(sub.pred_late_prob.between(0, 1).all(), "Probability outside [0,1]")
        elif task == "task2a":
            require((sub.pred_chilled_volume_m3 <= sub.pred_total_volume_m3 + 1e-7).all(), "Chilled exceeds total")
            m = sub.merge(data.read("task2a_test_inputs.csv"), on="row_id", validate="one_to_one")
            require(m.loc[m.brand.ne("Fresh"), "pred_chilled_volume_m3"].eq(0).all(), "Non-Fresh chilled forecast")
        else:
            validate_allocation(data, sub)
        report[task] = {"rows": len(sub), "columns": list(sub.columns), "valid": True}
    write_json(out / "reports/submission_validation.json", report)
    return report


def inference_task1(data, model_dir):
    x, _, _, meta = construct_task1(data, False)
    service = joblib.load(Path(model_dir) / "task1_service.joblib")
    lateness = joblib.load(Path(model_dir) / "task1_lateness.joblib")
    return pd.DataFrame({"delivery_id": meta.delivery_id, "pred_service_min": service.predict(x),
                         "pred_late_prob": np.clip(lateness.predict(x), 0, 1)})


def inference_task2a(data, model_dir):
    _, days, future, _ = demand_panel(data)
    total = joblib.load(Path(model_dir) / "task2a_total.joblib").predict(days)
    chilled = joblib.load(Path(model_dir) / "task2a_chilled.joblib").predict(days)
    keys = ["depot", "brand", "iso_year", "iso_week"]
    p = total[keys + ["prediction"]].rename(columns={"prediction": "pred_total_volume_m3"}).merge(
        chilled[keys + ["prediction"]].rename(columns={"prediction": "pred_chilled_volume_m3"}), on=keys, validate="one_to_one")
    p["pred_total_volume_m3"] = p.pred_total_volume_m3.clip(lower=0)
    p["pred_chilled_volume_m3"] = np.minimum(p.pred_chilled_volume_m3.clip(lower=0), p.pred_total_volume_m3)
    p.loc[p.brand.ne("Fresh"), "pred_chilled_volume_m3"] = 0
    result = future.merge(p, on=keys, validate="one_to_one")
    return result[["row_id", "depot", "brand", "iso_year", "iso_week", "pred_total_volume_m3", "pred_chilled_volume_m3"]]


def write_run_summary(data, out):
    """Display selected models and measured errors; never invent missing scores.

    Reads saved holdout metrics. Checks the CURRENT allocation CSV before
    reporting zero errors. This command does not fit or select any models.
    """
    from html import escape
    out = Path(out)
    report_dir = out / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    def read_report(name):
        path = report_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def chosen(target):
        selection = task1.get(target, {}).get("selection", [])
        if not selection:
            return "Not run"
        if len(selection) == 1:
            return selection[0][0]
        return " + ".join(f"{name} ({weight:.1%})" for name, weight in selection)

    def score(report, target, metric, unit=""):
        value = report.get(target, {}).get("holdout_metrics", {}).get(metric)
        if value is None:
            return "Not available"
        require(np.isfinite(float(value)), f"Invalid summary metric: {target}/{metric}")
        return f"{float(value):.3f}{unit}"

    task1, demand = read_report("task1_metrics.json"), read_report("task2a_metrics.json")
    served, validation_errors = "Not run", "Not checked"
    allocation = out / "submissions/submission_task2b.csv"
    if allocation.is_file():
        sub = pd.read_csv(allocation)
        if "decision" in sub:
            served = str(int(sub.decision.eq("served").sum()))
        try:
            validate_allocation(data, sub)
            validation_errors = "0"
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            # The validator stops at its first failure, so do not pretend this
            # number is an exhaustive count of every possible problem.
            validation_errors = "At least 1 - " + str(exc)
            LOG.error("Summary allocation validation failed: %s", exc)

    rows = [
        ("Task 1 chosen service model", chosen("service")),
        ("Task 1 service MAE", score(task1, "service", "mae", " minutes")),
        ("Task 1 chosen lateness model", chosen("lateness")),
        ("Task 1 lateness AUC", score(task1, "lateness", "roc_auc")),
        ("Task 2A total-volume blend MAE", score(demand, "total", "mae", " m3")),
        ("Task 2A chilled-volume blend MAE", score(demand, "chilled", "mae", " m3")),
        ("Task 2B served orders", served),
        ("Task 2B validation errors", validation_errors),
    ]
    summary = pd.DataFrame(rows, columns=["Metric", "Result"])
    note = ("Prediction metrics use the saved chronological holdouts; they are not hidden-test scores. "
            "Demand MAE is weekly volume in cubic metres. Chilled MAE uses Fresh rows only. "
            "Model percentages are ensemble weights. Allocation errors come from a fresh independent check of the current CSV; "
            "the organizer checker result is saved separately in official_checker.txt.")
    md = "# Run summary\n\n| Metric | Result |\n|---|---|\n"
    md += "\n".join("| " + label.replace("|", "\\|") + " | " + value.replace("|", "\\|").replace("\n", " ") + " |"
                    for label, value in rows)
    (report_dir / "run_summary.md").write_text(md + "\n\n" + note + "\n", encoding="utf-8")
    summary.to_csv(report_dir / "run_summary.csv", index=False)
    body = "\n".join(f"<tr><th scope='row'>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in rows)
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Waypoint run summary</title>
<style>
.waypoint-summary{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1160px;margin:24px auto;background:#191919;color:#f1f1f1;padding:24px;border-radius:12px}
.waypoint-summary h1{font-size:23px;font-weight:600;margin:0 0 6px}.waypoint-summary .sub{color:#b8b8b8;font-size:14px;margin:0 0 20px}
.waypoint-summary table{width:100%;border-collapse:collapse;table-layout:fixed}.waypoint-summary th,.waypoint-summary td{padding:17px 12px;border-top:1px solid #333;vertical-align:top;line-height:1.5;overflow-wrap:anywhere}
.waypoint-summary th{width:46%;text-align:left;font-size:17px;font-weight:400}.waypoint-summary td{font-family:ui-monospace,Consolas,monospace;font-size:16px}
.waypoint-summary .note{font-size:13px;line-height:1.7;color:#c2c2c2;margin:20px 12px 0}
@media(max-width:600px){.waypoint-summary{padding:14px;margin:8px}.waypoint-summary th,.waypoint-summary td{font-size:13px;padding:12px 6px}}
</style></head><body><section class="waypoint-summary"><h1>Waypoint Datathon</h1>
<p class="sub">Selected models and measured run results</p><table aria-label="Datathon results"><tbody>"""
    html += body + "</tbody></table><p class='note'>" + escape(note) + "</p></section></body></html>\n"
    (report_dir / "run_summary.html").write_text(html, encoding="utf-8")
    print("\n" + summary.to_string(index=False) + "\n\n" + note + "\n")
    return summary


def write_plots(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 130, "axes.labelcolor": "#22354b", "text.color": "#22354b"})
    p1 = out / "reports/task1_service_holdout.csv"
    if p1.exists() and (out / "reports/task1_lateness_holdout.csv").exists():
        p = pd.read_csv(p1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
        axes[0].hexbin(p.actual, p.prediction, gridsize=40, mincnt=1, cmap="Blues")
        lim = max(p.actual.max(), p.prediction.max())
        axes[0].plot([0, lim], [0, lim], color="#d45e39", lw=1)
        axes[0].set(xlabel="Actual service time (min)", ylabel="Predicted service time (min)", title="Service: untouched six-week holdout")
        p = pd.read_csv(out / "reports/task1_lateness_holdout.csv")
        p["bin"] = pd.cut(p.prediction, np.linspace(0, 1, 11), include_lowest=True)
        c = p.groupby("bin", observed=True).agg(predicted=("prediction", "mean"), actual=("actual", "mean"), n=("actual", "size"))
        axes[1].plot([0, 1], [0, 1], color="#bac2cb", ls="--")
        axes[1].plot(c.predicted, c.actual, "o-", color="#14857a")
        axes[1].set(xlabel="Mean predicted probability", ylabel="Observed late fraction", title="Lateness calibration", xlim=(0, 1), ylim=(0, 1))
        fig.savefig(out / "reports/task1_validation.png")
        plt.close(fig)
    p2 = out / "reports/task2a_holdout.csv"
    if p2.exists():
        p = pd.read_csv(p2)
        fig, axes = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)
        for ax, ((depot, brand), g) in zip(axes.flat, p.groupby(["depot", "brand"])):
            g = g.sort_values(["iso_year", "iso_week"])
            x = np.arange(len(g))
            ax.plot(x, g.total, "o-", color="#22354b", label="Observed")
            ax.plot(x, g.pred_total_volume_m3, "s--", color="#14857a", label="Forecast")
            ax.set(title=f"{depot} / {brand}", xlabel="Week within ten-week holdout", ylabel="Total volume (m3)")
            ax.legend(frameon=False, fontsize=8)
        fig.savefig(out / "reports/task2a_validation.png")
        plt.close(fig)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", default="data", help="Root of the extracted organizer data folder")
    parser.add_argument("--output", default="outputs", help="Output directory for models, reports and submissions")
    parser.add_argument("--task", choices=["all", "task1", "task2a", "task2b", "validate", "predict", "summary"], default="all",
                        help="Use summary to display saved model/metric results without retraining")
    parser.add_argument("--preset", choices=["quick", "balanced", "thorough"], default="balanced",
                        help="Fixed candidate configurations; thorough compares 13 configurations per Task1 target")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--allocator-seconds", type=float, default=120, help="Time limit for EACH of the two allocation policies")
    parser.add_argument("--checker", default="check_allocation.py", help="Unmodified organizer checker")
    parser.add_argument("--priority-config", default=None, help="Optional JSON overrides for documented allocation weights")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    global SEED
    SEED = args.seed
    require(args.jobs > 0 and args.allocator_seconds > 0, "Jobs and time limit must be positive")
    np.random.seed(SEED)
    out = Path(args.output).expanduser().resolve()
    for folder in [out, out / "models", out / "reports", out / "submissions"]:
        folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log", encoding="utf-8")])
    data = Data(args.data)
    write_json(out / "reports/data_manifest.json", data.manifest())
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "lightgbm", "xgboost", "ortools", "joblib"]:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    write_json(out / f"reports/run_config_{args.task}.json", {**vars(args), "python": platform.python_version(), "packages": versions})
    with threadpool_limits(limits=args.jobs):
        if args.task in {"all", "task1"}:
            run_task1(data, out, args)
        if args.task in {"all", "task2a"}:
            run_task2a(data, out, args)
        if args.task in {"all", "task2b"}:
            run_task2b(data, out, args)
        if args.task in {"all", "validate"}:
            LOG.info("SUBMISSION CHECKS: %s", validate_submissions(data, out))
        if args.task == "predict":
            print("Task1 input sample:\n", data.read("task1_test_inputs.csv").head().to_string(index=False))
            print("Task1 saved-model predictions:\n", inference_task1(data, out / "models").head().to_string(index=False))
            print("Task2A input sample:\n", data.read("task2a_test_inputs.csv").head().to_string(index=False))
            print("Task2A saved-model predictions:\n", inference_task2a(data, out / "models").head().to_string(index=False))
        if args.task != "summary":
            write_plots(out)
        write_run_summary(data, out)
    LOG.info("Finished. Files: %s", out)


if __name__ == "__main__":
    # Keep pickle class paths importable in Python, notebooks and fresh processes.
    from waypoint_datathon import main as imported_main
    imported_main()
