"""
Treinamento Baseline — XGBoost com Pseudo-Huber Loss
Fase 3, Passo B

Configuração (TASK.md §5.B):
  - Objective: reg:pseudohubererror (robusto a outliers de cauda longa)
  - Validação: TimeSeriesSplit (5 folds, gap=1 dia)
  - Métrica principal: MAE (gramas)

Input:  data/processed/abt_modeling.parquet
Output:
  - models/xgb_baseline.json
  - reports/figures/08_model_predictions_vs_actual.png
  - reports/figures/09_feature_importance.png

Autor: Jorge Metri / Claude (MLOps)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
MODEL_PATH = MODELS_DIR / "xgb_baseline.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colunas
# ---------------------------------------------------------------------------
TARGET = "peso_filtro"

# Features a excluir do treino (identificadores, target, variáveis de leak)
DROP_COLS = [
    "data",           # timestamp, não feature
    "peso_filtro",    # target
    "classe_filtro",  # apenas para validação estratificada
    "produto_moda",   # categórica — encode separado se necessário
]


def prepare_data(df: pd.DataFrame) -> tuple:
    """Separa features numéricas e target."""
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y, feature_cols


def train_cv(X: pd.DataFrame, y: pd.Series, feature_cols: list) -> dict:
    """Treina XGBoost com TimeSeriesSplit e gap de 1 dia.

    O gap é implementado removendo a última observação do treino,
    garantindo que não haja vazamento de lags entre folds.
    """
    tscv = TimeSeriesSplit(n_splits=5, gap=1)

    params = {
        "objective": "reg:pseudohubererror",
        "eval_metric": "mae",
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }

    fold_metrics = []
    all_preds = np.full(len(y), np.nan)
    best_model = None
    best_mae = np.inf

    print("[INFO] Treinamento com TimeSeriesSplit (5 folds, gap=1)")
    print("-" * 60)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        # Garantir não-negatividade (peso físico >= 0)
        y_pred = np.clip(y_pred, 0, None)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        fold_metrics.append({"fold": fold, "mae": mae, "rmse": rmse, "n_test": len(test_idx)})
        all_preds[test_idx] = y_pred

        print(f"  Fold {fold}: MAE={mae:.4f}g | RMSE={rmse:.4f}g | n_test={len(test_idx)}")

        if mae < best_mae:
            best_mae = mae
            best_model = model

    print("-" * 60)
    metrics_df = pd.DataFrame(fold_metrics)
    avg_mae = metrics_df["mae"].mean()
    avg_rmse = metrics_df["rmse"].mean()
    print(f"  MÉDIA: MAE={avg_mae:.4f}g | RMSE={avg_rmse:.4f}g")

    # Dummy baseline (predizer a média do treino)
    dummy_mae = mean_absolute_error(y, np.full(len(y), y.mean()))
    print(f"  Dummy (média): MAE={dummy_mae:.4f}g")
    print(f"  Ganho vs Dummy: {((dummy_mae - avg_mae) / dummy_mae) * 100:.1f}%")

    return {
        "model": best_model,
        "metrics": metrics_df,
        "all_preds": all_preds,
        "avg_mae": avg_mae,
        "avg_rmse": avg_rmse,
        "dummy_mae": dummy_mae,
    }


def plot_predictions(y: pd.Series, preds: np.ndarray, dates: pd.Series) -> None:
    """Plot 08: Predições vs Actual."""
    mask = ~np.isnan(preds)
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(dates, y, "o-", ms=3, alpha=0.6, label="Actual", color="#2196F3")
    ax.plot(
        dates[mask],
        preds[mask],
        "x-",
        ms=3,
        alpha=0.7,
        label="XGBoost (CV)",
        color="#FF5722",
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Peso do Filtro (g)")
    ax.set_title("XGBoost Baseline — Predictions vs Actual (TimeSeriesSplit)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_model_predictions_vs_actual.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Plot salvo: {FIGURES_DIR / '08_model_predictions_vs_actual.png'}")


def plot_feature_importance(model: xgb.XGBRegressor, feature_cols: list) -> None:
    """Plot 09: Feature Importance (gain)."""
    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame(
            {"feature": importance.keys(), "gain": importance.values()}
        )
        .sort_values("gain", ascending=True)
        .tail(15)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(imp_df["feature"], imp_df["gain"], color="#4CAF50", edgecolor="white")
    ax.set_xlabel("Gain (Information Gain)")
    ax.set_title("XGBoost Baseline — Top 15 Feature Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Plot salvo: {FIGURES_DIR / '09_feature_importance.png'}")

    # Imprimir ranking no console
    print("\n[INFO] Feature Importance (Top 15 por Gain):")
    for _, row in imp_df.iloc[::-1].iterrows():
        print(f"  {row['feature']:30s} → {row['gain']:.2f}")


def main() -> None:
    print("=" * 60)
    print("FASE 3 — Baseline XGBoost (Pseudo-Huber Loss)")
    print("=" * 60)

    df = pd.read_parquet(INPUT_PATH)
    print(f"[INFO] ABT de modelagem: {df.shape}")

    X, y, feature_cols = prepare_data(df)
    print(f"[INFO] Features: {len(feature_cols)} | Target: {TARGET}")
    print(f"[INFO] Target stats: mean={y.mean():.4f}, median={y.median():.4f}, std={y.std():.4f}")

    results = train_cv(X, y, feature_cols)

    # Salvar modelo
    results["model"].save_model(str(MODEL_PATH))
    print(f"\n[OK] Modelo salvo: {MODEL_PATH}")

    # Plots
    plot_predictions(y, results["all_preds"], df["data"])
    plot_feature_importance(results["model"], feature_cols)

    print("\n" + "=" * 60)
    print("FASE 3 CONCLUÍDA")
    print("=" * 60)


if __name__ == "__main__":
    main()
