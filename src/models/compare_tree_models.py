"""
Benchmarking de Modelos de Árvore — RF, LightGBM vs XGBoost Baseline
Fase 3, Passo B.2

Compara MAE médio via TimeSeriesSplit(n_splits=5, gap=1) para decidir
se vale tunar árvores ou avançar direto para a KAN (TASK.md §5.B.2).

Input:  data/processed/abt_modeling.parquet
Output: reports/figures/10_tree_benchmark.png  (barplot comparativo)

Autor: Jorge Metri / Claude (MLOps)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("reports/figures")
INPUT_PATH = PROCESSED_DIR / "abt_modeling.parquet"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colunas
# ---------------------------------------------------------------------------
TARGET = "peso_filtro"
DROP_COLS = [
    "data",
    "peso_filtro",
    "classe_filtro",
    "produto_moda",
]


def prepare_data(df: pd.DataFrame) -> tuple:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y, feature_cols


def define_models() -> dict:
    """Retorna dicionário {nome: modelo} com configs comparáveis."""
    return {
        "XGBoost (Huber)": xgb.XGBRegressor(
            objective="reg:pseudohubererror",
            eval_metric="mae",
            max_depth=5,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM (Huber)": lgb.LGBMRegressor(
            objective="huber",
            metric="mae",
            max_depth=5,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
        ),
    }


def evaluate_model(
    name: str, model, X: pd.DataFrame, y: pd.Series
) -> dict:
    """Avalia um modelo com TimeSeriesSplit e retorna métricas por fold."""
    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    fold_mae, fold_rmse = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit com early stopping para XGB e LGBM; RF usa fit simples
        if isinstance(model, xgb.XGBRegressor):
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        elif isinstance(model, lgb.LGBMRegressor):
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            model.fit(X_train, y_train)

        y_pred = np.clip(model.predict(X_test), 0, None)
        fold_mae.append(mean_absolute_error(y_test, y_pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        "name": name,
        "mae_folds": fold_mae,
        "rmse_folds": fold_rmse,
        "mae_mean": np.mean(fold_mae),
        "mae_std": np.std(fold_mae),
        "rmse_mean": np.mean(fold_rmse),
        "rmse_std": np.std(fold_rmse),
    }


def plot_benchmark(results: list[dict]) -> None:
    """Barplot comparativo de MAE médio ± std."""
    names = [r["name"] for r in results]
    maes = [r["mae_mean"] for r in results]
    stds = [r["mae_std"] for r in results]

    colors = ["#FF5722", "#2196F3", "#4CAF50"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, maes, yerr=stds, capsize=8, color=colors,
                  edgecolor="white", linewidth=1.5)

    # Anotar valores sobre as barras
    for bar, mae, std in zip(bars, maes, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.001,
            f"{mae:.4f}g",
            ha="center", va="bottom", fontweight="bold", fontsize=12,
        )

    # Linha de referência do dummy
    ax.axhline(y=0.0634, color="gray", linestyle="--", alpha=0.7, label="Dummy (média)")
    ax.set_ylabel("MAE (g)")
    ax.set_title("Benchmark — Modelos de Árvore (TimeSeriesSplit, 5 folds, gap=1)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(maes) + max(stds) + 0.015)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_tree_benchmark.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Plot salvo: {FIGURES_DIR / '10_tree_benchmark.png'}")


def main() -> None:
    print("=" * 65)
    print("BENCHMARKING — Modelos de Árvore (RF, LightGBM vs XGBoost)")
    print("=" * 65)

    df = pd.read_parquet(INPUT_PATH)
    X, y, feature_cols = prepare_data(df)
    print(f"[INFO] ABT: {df.shape} | Features: {len(feature_cols)}")
    print(f"[INFO] Target: mean={y.mean():.4f}, median={y.median():.4f}\n")

    models = define_models()
    results = []

    for name, model in models.items():
        print(f"[TRAIN] {name}...")
        res = evaluate_model(name, model, X, y)
        results.append(res)
        print(f"         MAE = {res['mae_mean']:.4f} ± {res['mae_std']:.4f}g")
        print(f"        RMSE = {res['rmse_mean']:.4f} ± {res['rmse_std']:.4f}g")
        folds_str = " | ".join(f"F{i+1}={m:.4f}" for i, m in enumerate(res["mae_folds"]))
        print(f"        [{folds_str}]\n")

    # Tabela comparativa
    print("-" * 65)
    print(f"{'Modelo':<22s}  {'MAE':>10s}  {'± Std':>8s}  {'RMSE':>10s}  {'vs Dummy':>10s}")
    print("-" * 65)
    dummy_mae = 0.0634
    for r in sorted(results, key=lambda x: x["mae_mean"]):
        gain = ((dummy_mae - r["mae_mean"]) / dummy_mae) * 100
        print(
            f"{r['name']:<22s}  {r['mae_mean']:>10.4f}  {r['mae_std']:>8.4f}  "
            f"{r['rmse_mean']:>10.4f}  {gain:>+9.1f}%"
        )
    print("-" * 65)

    # Decisão automática
    best = min(results, key=lambda x: x["mae_mean"])
    xgb_res = next(r for r in results if "XGBoost" in r["name"])
    delta = xgb_res["mae_mean"] - best["mae_mean"]

    print(f"\n[DECISÃO] Melhor modelo: {best['name']} (MAE={best['mae_mean']:.4f}g)")
    if best["name"] != xgb_res["name"] and delta > 0.005:
        print(f"[DECISÃO] Ganho significativo de {delta:.4f}g vs XGBoost → TUNAR {best['name']}")
    else:
        print(f"[DECISÃO] Diferença marginal ({delta:.4f}g) → Avançar para KAN/MLP")

    plot_benchmark(results)

    print("\n" + "=" * 65)
    print("BENCHMARKING CONCLUÍDO")
    print("=" * 65)


if __name__ == "__main__":
    main()
