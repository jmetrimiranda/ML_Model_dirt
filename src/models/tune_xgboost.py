"""
Fine-Tuning XGBoost — Optuna (100 trials)
Fase 3.D (TASK.md §7)

Busca exaustiva de hiperparâmetros mantendo:
  - objective='reg:pseudohubererror' (mandatório)
  - TimeSeriesSplit(n_splits=5, gap=1)
  - Métrica: MAE médio dos 5 folds

Output:
  - models/xgb_optimized.json
  - reports/figures/13_optuna_optimization.png
  - reports/figures/14_optimized_predictions.png
  - reports/figures/15_optimized_feature_importance.png

Autor: Jorge Metri / Claude (MLOps)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
INPUT_PATH = PROCESSED_DIR / "abt_modeling.parquet"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET = "peso_filtro"
DROP_COLS = ["data", "peso_filtro", "classe_filtro", "produto_moda"]
N_TRIALS = 100
SEED = 42


def load_data() -> tuple:
    df = pd.read_parquet(INPUT_PATH)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET]
    return X, y, feature_cols, df


# ---------------------------------------------------------------------------
# Optuna Objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """Minimizar MAE médio via TimeSeriesSplit."""
    X, y, _, _ = load_data()

    params = {
        "objective": "reg:pseudohubererror",
        "eval_metric": "mae",
        "verbosity": 0,
        "random_state": SEED,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "early_stopping_rounds": 50,
    }

    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    fold_maes = []

    for train_idx, test_idx in tscv.split(X):
        model = xgb.XGBRegressor(**params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[test_idx], y.iloc[test_idx])],
            verbose=False,
        )
        preds = np.clip(model.predict(X.iloc[test_idx]), 0, None)
        fold_maes.append(mean_absolute_error(y.iloc[test_idx], preds))

    return np.mean(fold_maes)


# ---------------------------------------------------------------------------
# Train Final Model
# ---------------------------------------------------------------------------
def train_final(best_params: dict) -> dict:
    """Treina modelo final com melhores params e coleta métricas por fold."""
    X, y, feature_cols, df = load_data()

    full_params = {
        "objective": "reg:pseudohubererror",
        "eval_metric": "mae",
        "verbosity": 0,
        "random_state": SEED,
        "early_stopping_rounds": 50,
        **best_params,
    }

    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    fold_metrics = []
    all_preds = np.full(len(y), np.nan)
    best_model = None
    best_mae = np.inf

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        model = xgb.XGBRegressor(**full_params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[test_idx], y.iloc[test_idx])],
            verbose=False,
        )
        preds = np.clip(model.predict(X.iloc[test_idx]), 0, None)
        mae = mean_absolute_error(y.iloc[test_idx], preds)
        rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], preds))

        fold_metrics.append({"fold": fold, "mae": mae, "rmse": rmse})
        all_preds[test_idx] = preds

        if mae < best_mae:
            best_mae = mae
            best_model = model

        print(f"  Fold {fold}: MAE={mae:.4f}g | RMSE={rmse:.4f}g")

    metrics_df = pd.DataFrame(fold_metrics)
    avg_mae = metrics_df["mae"].mean()
    avg_rmse = metrics_df["rmse"].mean()

    return {
        "model": best_model,
        "metrics": metrics_df,
        "all_preds": all_preds,
        "avg_mae": avg_mae,
        "avg_rmse": avg_rmse,
        "dates": df["data"].values,
        "y": y.values,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_optimization_history(study: optuna.Study) -> None:
    """Plot 13: Optimization history."""
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    best_so_far = [min(values[: i + 1]) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(range(len(values)), values, alpha=0.4, s=20, color="#2196F3",
               label="Trial MAE")
    ax.plot(best_so_far, color="#FF5722", linewidth=2, label="Best MAE")
    ax.axhline(y=0.0499, color="gray", linestyle="--", alpha=0.7,
               label="Baseline (0.0499g)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("MAE (g)")
    ax.set_title(f"Optuna — Optimization History ({len(values)} trials)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_optuna_optimization.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Plot: {FIGURES_DIR / '13_optuna_optimization.png'}")


def plot_predictions(results: dict) -> None:
    """Plot 14: Optimized predictions vs actual."""
    dates = pd.to_datetime(results["dates"])
    y = results["y"]
    preds = results["all_preds"]
    mask = ~np.isnan(preds)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y, "o-", ms=3, alpha=0.6, label="Actual", color="#2196F3")
    ax.plot(dates[mask], preds[mask], "x-", ms=3, alpha=0.7,
            label=f"XGBoost Optimized (MAE={results['avg_mae']:.4f}g)", color="#4CAF50")
    ax.set_xlabel("Data")
    ax.set_ylabel("Peso do Filtro (g)")
    ax.set_title("XGBoost Optimized — Predictions vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "14_optimized_predictions.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Plot: {FIGURES_DIR / '14_optimized_predictions.png'}")


def plot_feature_importance(model: xgb.XGBRegressor) -> None:
    """Plot 15: Feature importance (optimized model)."""
    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame({"feature": importance.keys(), "gain": importance.values()})
        .sort_values("gain", ascending=True)
        .tail(15)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(imp_df["feature"], imp_df["gain"], color="#4CAF50", edgecolor="white")
    ax.set_xlabel("Gain")
    ax.set_title("XGBoost Optimized — Top 15 Feature Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "15_optimized_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Plot: {FIGURES_DIR / '15_optimized_feature_importance.png'}")

    print("\n[INFO] Feature Importance (Top 15):")
    for _, row in imp_df.iloc[::-1].iterrows():
        print(f"  {row['feature']:30s} → {row['gain']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 65)
    print("FASE 3.D — Fine-Tuning XGBoost com Optuna")
    print("=" * 65)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # --- Otimização ---
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="xgb_dust_prediction",
    )

    print(f"[INFO] Iniciando {N_TRIALS} trials...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n[INFO] Melhor MAE: {study.best_value:.4f}g")
    print(f"[INFO] Melhores parâmetros:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # --- Treinar modelo final ---
    print(f"\n[INFO] Treinando modelo final com melhores parâmetros...")
    print("-" * 65)
    results = train_final(study.best_params)
    print("-" * 65)

    avg_mae = results["avg_mae"]
    avg_rmse = results["avg_rmse"]
    baseline_mae = 0.0499
    dummy_mae = 0.0634

    print(f"\n  RESULTADO FINAL:")
    print(f"  XGBoost Optimized MAE: {avg_mae:.4f}g | RMSE: {avg_rmse:.4f}g")
    print(f"  XGBoost Baseline MAE:  {baseline_mae:.4f}g")
    print(f"  Dummy MAE:             {dummy_mae:.4f}g")
    print(f"  Ganho vs Baseline:     {((baseline_mae - avg_mae) / baseline_mae) * 100:+.1f}%")
    print(f"  Ganho vs Dummy:        {((dummy_mae - avg_mae) / dummy_mae) * 100:+.1f}%")

    # --- Salvar modelo ---
    results["model"].save_model(str(MODELS_DIR / "xgb_optimized.json"))
    print(f"\n[OK] Modelo salvo: {MODELS_DIR / 'xgb_optimized.json'}")

    # --- Plots ---
    plot_optimization_history(study)
    plot_predictions(results)
    plot_feature_importance(results["model"])

    print("\n" + "=" * 65)
    print("FASE 3.D CONCLUÍDA")
    print("=" * 65)


if __name__ == "__main__":
    main()
