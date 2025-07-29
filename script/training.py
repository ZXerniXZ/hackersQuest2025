#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py – CatBoost tuner + CV + report
"""

import json, argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
import optuna

TARGET = "peso_adulto_kg"              # colonna da predire
CAT_COLS = ["razza", "sesso", "tipo_cibo", "pelo_lungo", "sterilizzato"]

# ----------------------------------------------------------------------------- #
def trim_outliers(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Mantiene la frazione centrale di distribuzione sul target.
       • 0  o None ⇒ nessun taglio
       • 0<frac≤1 ⇒ frazione (es. 0.8 = 80 %)
       • frac>1   ⇒ trattato come percentuale (80 ⇒ 0.8) """
    if not frac or frac <= 0:
        return df
    if frac > 1:
        frac = frac / 100.0
    frac = max(0.0, min(frac, 1.0))
    lo, hi = 0.5 - frac / 2, 0.5 + frac / 2
    mask = df[TARGET].between(*df[TARGET].quantile([lo, hi]))
    print(f"Trim outliers: {len(df) - mask.sum()} rimossi / {len(df)}")
    return df.loc[mask].reset_index(drop=True)

# ----------------------------------------------------------------------------- #
def evaluate(y_true, y_pred):
    """RMSE, MAE, R² (compatibile con vecchie sklearn)."""
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ----------------------------------------------------------------------------- #
def cv_predictions(params, X, y, cat_idx, cv):
    model = CatBoostRegressor(**params, cat_features=cat_idx)
    return cross_val_predict(model, X, y, cv=cv, n_jobs=1, method="predict")

# ----------------------------------------------------------------------------- #
def main(a):
    # --- lettura dati -------------------------------------------------------- #
    try:
        train_df = pd.read_csv(a.train)
        test_df  = pd.read_csv(a.test)
    except UnicodeDecodeError:
        train_df = pd.read_csv(a.train, encoding="latin-1")
        test_df  = pd.read_csv(a.test,  encoding="latin-1")

    train_df = trim_outliers(train_df, a.trim_outliers)

    # fill NA categoriche per CatBoost
    for c in CAT_COLS:
        for df in (train_df, test_df):
            if c in df.columns:
                df[c] = df[c].astype(str).fillna("missing")

    X, y      = train_df.drop(TARGET, axis=1), train_df[TARGET]
    X_test    = test_df.copy()
    cat_idx   = [X.columns.get_loc(c) for c in CAT_COLS if c in X.columns]
    y_log     = np.log1p(y)                     # stabilizza la varianza
    cv        = KFold(n_splits=a.folds, shuffle=True, random_state=42)

    # --- Optuna ---------------------------------------------------------------- #
    def objective(trial):
        p = dict(
            learning_rate  = trial.suggest_float("lr", 5e-3, 5e-2, log=True),
            depth          = trial.suggest_int("depth", 4, 6),
            iterations     = trial.suggest_int("iterations", 800, 2000),
            l2_leaf_reg    = trial.suggest_float("l2", 0.1, 10, log=True),
            subsample      = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bylevel = trial.suggest_float("csl", 0.6, 1.0),
            bagging_temperature = trial.suggest_float("temp", 0.0, 1.0),
            loss_function  = "RMSE",
            verbose        = False,
            use_best_model = False
        )
        preds = cv_predictions(p, X, y_log, cat_idx, cv)
        rmse, _, _ = evaluate(y_log, preds)
        return rmse

    if a.trials > 0:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=a.trials, show_progress_bar=True)
        best = study.best_params
    else:
        # parametri di default se nessuna prova
        best = dict(lr=0.01, depth=5, iterations=1500, l2=1.0,
                    subsample=0.8, csl=0.8, temp=0.5)

    # mappatura abbreviazioni -> nomi CatBoost
    keymap = {"lr":"learning_rate", "l2":"l2_leaf_reg", "csl":"colsample_bylevel",
              "temp":"bagging_temperature"}
    best   = {keymap.get(k, k):v for k,v in best.items()}
    best.update(loss_function="RMSE", verbose=False, use_best_model=False)

    # -------- training finale + ensemble di semi ----------------------------- #
    seeds = range(a.ensemble_seeds)
    preds_cv, preds_test = [], []

    for s in seeds:
        m = CatBoostRegressor(**best, random_seed=s, cat_features=cat_idx)
        preds_cv.append( cv_predictions(best | {"random_seed":s}, X, y_log, cat_idx, cv) )
        m.fit(X, y_log)
        preds_test.append( m.predict(X_test) )

    pred_cv_mean   = np.mean(preds_cv,   axis=0)
    pred_test_mean = np.mean(preds_test, axis=0)

    rmse, mae, r2  = evaluate(y_log, pred_cv_mean)
    print(f"★ CV  (log‑target)  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.3f}")

    # back‑transform
    submission = pd.DataFrame({
        "Id": np.arange(len(pred_test_mean)),
        TARGET: np.expm1(pred_test_mean)
    })
    sub_path = "submission_cat_weight.csv"
    submission.to_csv(sub_path, index=False)
    print(f"File '{sub_path}' creato.")

    # ------- report JSON ----------------------------------------------------- #
    report = {
        "best_params": best,
        "cv_rmse":     float(rmse),
        "cv_mae":      float(mae),
        "cv_r2":       float(r2),
        "ensemble_seeds": list(seeds),
        "files": {"submission": sub_path}
    }
    with open("training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report salvato → training_report.json")

# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test",  required=True)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--folds",  type=int, default=5)
    p.add_argument("--ensemble_seeds", type=int, default=1)
    p.add_argument("--trim_outliers",  type=float, default=0.0,
                   help="frazione (0‑1) o percentuale (>1) di distribuzione da tenere")
    main(p.parse_args())
