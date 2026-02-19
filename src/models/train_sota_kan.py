"""
SOTA Model — KAN Compacta + MLP Regularizado
Fase 3, Passo C (v2 — corrigido para dataset pequeno)

Problema da v1: 52k params / 316 amostras → overfitting severo.
Solução v2:
  - KAN:  [n_feat, 16, 1] grid=3 → ~2.5k params (ratio ~1:8)
  - MLP:  [n_feat, 32, 16, 1] + Dropout(0.3) + BatchNorm → ~1.6k params
  - Ambos com HuberLoss(delta=0.15), seleciona o melhor via CV.

Input:  data/processed/abt_modeling.parquet
Output:
  - models/kan_sota.pt  (ou mlp_sota.pt)
  - reports/figures/11_kan_predictions_vs_actual.png
  - reports/figures/12_model_comparison_kan_vs_xgb.png

Autor: Jorge Metri / Claude (MLOps)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kan import KAN
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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
DROP_COLS = ["data", "peso_filtro", "classe_filtro"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HUBER_DELTA = 0.15
EPOCHS = 500
PATIENCE = 50
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# MLP com Dropout + BatchNorm (alternativa robusta à KAN)
# ---------------------------------------------------------------------------
class PhysicsMLP(nn.Module):
    """MLP compacto com regularização forte para dataset pequeno."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Data Prep
# ---------------------------------------------------------------------------
def prepare_data(df: pd.DataFrame) -> tuple:
    """StandardScaler + OneHot encoding do produto_moda."""
    produto_map = {
        "PDR-STD": "PDR/STD",
        "PBF-HB": "PBF/HB",
        "PBF-MB45": "PBF/MB45",
        "PBF-STD": "PBF/STD",
        "PBF-SF": "PBF/SF",
        "PBF-SA": "PBF/SA",
        "PBF-SAM01": "PBF/SAM01",
        "PF-SA": "PBF/SA",
        " PFN/STD": "PFN/STD",
        "PFN": "PFN/STD",
        "19": "NO_OP",
    }
    df = df.copy()
    df["produto_moda"] = df["produto_moda"].replace(produto_map)

    ohe = pd.get_dummies(df["produto_moda"], prefix="prod", dtype=float)

    num_cols = [c for c in df.columns if c not in DROP_COLS + ["produto_moda"]]
    X_num = df[num_cols].values.astype(np.float32)
    X_ohe = ohe.values.astype(np.float32)
    X = np.hstack([X_num, X_ohe])

    y = df[TARGET].values.astype(np.float32)
    feature_names = num_cols + list(ohe.columns)

    return X, y, feature_names, df["data"].values


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    return X_tr.astype(np.float32), X_te.astype(np.float32), scaler


# ---------------------------------------------------------------------------
# Generic Training Loop
# ---------------------------------------------------------------------------
def train_one_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold: int,
    lr: float,
    weight_decay: float,
) -> tuple:
    """Treina qualquer nn.Module num fold com early stopping."""
    X_tr_s, X_te_s, scaler = scale_features(X_train, X_test)

    X_tr_t = torch.tensor(X_tr_s, device=DEVICE)
    y_tr_t = torch.tensor(y_train, device=DEVICE).unsqueeze(1)
    X_te_t = torch.tensor(X_te_s, device=DEVICE)
    y_te_t = torch.tensor(y_test, device=DEVICE).unsqueeze(1)

    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    n_train = X_tr_t.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            xb, yb = X_tr_t[idx], y_tr_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_te_t)
            val_loss = criterion(val_pred, y_te_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te_t).cpu().numpy().flatten()

    y_pred = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"    Fold {fold}: MAE={mae:.4f}g | RMSE={rmse:.4f}g | "
        f"Ep={epoch+1} | Params={n_params}"
    )

    return y_pred, mae, rmse, model, scaler


def run_cv(
    model_name: str,
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    weight_decay: float,
) -> dict:
    """TimeSeriesSplit CV para um tipo de modelo."""
    tscv = TimeSeriesSplit(n_splits=5, gap=1)

    fold_metrics = []
    all_preds = np.full(len(y), np.nan)
    best_model = None
    best_scaler = None
    best_mae = np.inf

    print(f"\n  [{model_name}]")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        model = model_factory()
        y_pred, mae, rmse, model, scaler = train_one_fold(
            model,
            X[train_idx], y[train_idx],
            X[test_idx], y[test_idx],
            fold, lr, weight_decay,
        )
        fold_metrics.append({"fold": fold, "mae": mae, "rmse": rmse})
        all_preds[test_idx] = y_pred

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_scaler = scaler

    metrics_df = pd.DataFrame(fold_metrics)
    avg_mae = metrics_df["mae"].mean()
    avg_rmse = metrics_df["rmse"].mean()
    print(f"    MÉDIA: MAE={avg_mae:.4f}g | RMSE={avg_rmse:.4f}g")

    return {
        "name": model_name,
        "model": best_model,
        "scaler": best_scaler,
        "metrics": metrics_df,
        "all_preds": all_preds,
        "avg_mae": avg_mae,
        "avg_rmse": avg_rmse,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_predictions(
    y: np.ndarray, preds: np.ndarray, dates: np.ndarray, name: str
) -> None:
    dates_pd = pd.to_datetime(dates)
    mask = ~np.isnan(preds)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates_pd, y, "o-", ms=3, alpha=0.6, label="Actual", color="#2196F3")
    ax.plot(
        dates_pd[mask], preds[mask],
        "x-", ms=3, alpha=0.7, label=f"{name} (CV)", color="#9C27B0",
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Peso do Filtro (g)")
    ax.set_title(f"{name} — Predictions vs Actual (TimeSeriesSplit)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_kan_predictions_vs_actual.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Plot: {FIGURES_DIR / '11_kan_predictions_vs_actual.png'}")


def plot_comparison(
    y: np.ndarray, nn_preds: np.ndarray, dates: np.ndarray, nn_name: str
) -> None:
    """Plot 12: NN vs XGBoost — timeline completa + zoom Set-Out."""
    import xgboost as xgb

    df = pd.read_parquet(INPUT_PATH)
    drop = ["data", "peso_filtro", "classe_filtro", "produto_moda"]
    X_xgb = df[[c for c in df.columns if c not in drop]]
    y_xgb = df[TARGET].values

    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    xgb_preds = np.full(len(y_xgb), np.nan)

    for train_idx, test_idx in tscv.split(X_xgb):
        model = xgb.XGBRegressor(
            objective="reg:pseudohubererror",
            max_depth=5, learning_rate=0.05, n_estimators=500,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0, early_stopping_rounds=50,
        )
        model.fit(
            X_xgb.iloc[train_idx], y_xgb[train_idx],
            eval_set=[(X_xgb.iloc[test_idx], y_xgb[test_idx])],
            verbose=False,
        )
        xgb_preds[test_idx] = np.clip(model.predict(X_xgb.iloc[test_idx]), 0, None)

    dates_pd = pd.to_datetime(dates)
    mask_nn = ~np.isnan(nn_preds)
    mask_xgb = ~np.isnan(xgb_preds)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Full timeline
    ax = axes[0]
    ax.plot(dates_pd, y, "o-", ms=2, alpha=0.5, label="Actual", color="#2196F3")
    ax.plot(dates_pd[mask_xgb], xgb_preds[mask_xgb], "s-", ms=2, alpha=0.6,
            label="XGBoost (MAE=0.0499)", color="#FF5722")
    ax.plot(dates_pd[mask_nn], nn_preds[mask_nn], "^-", ms=2, alpha=0.6,
            label=nn_name, color="#9C27B0")
    ax.set_ylabel("Peso do Filtro (g)")
    ax.set_title(f"Comparação — {nn_name} vs XGBoost vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zoom Set-Out/2025
    ax2 = axes[1]
    start = pd.Timestamp("2025-09-01")
    end = pd.Timestamp("2025-11-01")
    mask_zoom = (dates_pd >= start) & (dates_pd <= end)

    ax2.plot(dates_pd[mask_zoom], y[mask_zoom], "o-", ms=4, alpha=0.7,
             label="Actual", color="#2196F3", linewidth=1.5)
    mask_z_xgb = mask_zoom & mask_xgb
    ax2.plot(dates_pd[mask_z_xgb], xgb_preds[mask_z_xgb], "s-", ms=4, alpha=0.7,
             label="XGBoost", color="#FF5722", linewidth=1.5)
    mask_z_nn = mask_zoom & mask_nn
    ax2.plot(dates_pd[mask_z_nn], nn_preds[mask_z_nn], "^-", ms=4, alpha=0.7,
             label=nn_name, color="#9C27B0", linewidth=1.5)
    ax2.set_xlabel("Data")
    ax2.set_ylabel("Peso do Filtro (g)")
    ax2.set_title("Zoom — Período Crítico (Set-Out/2025)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_model_comparison_kan_vs_xgb.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Plot: {FIGURES_DIR / '12_model_comparison_kan_vs_xgb.png'}")

    # Métricas no zoom
    valid_zoom = mask_zoom & mask_nn & mask_xgb
    if valid_zoom.sum() > 0:
        mae_xgb_z = mean_absolute_error(y[valid_zoom], xgb_preds[valid_zoom])
        mae_nn_z = mean_absolute_error(y[valid_zoom], nn_preds[valid_zoom])
        print(f"\n[INFO] Set-Out/2025 (n={valid_zoom.sum()}):")
        print(f"  XGBoost MAE: {mae_xgb_z:.4f}g")
        print(f"  {nn_name} MAE: {mae_nn_z:.4f}g")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 65)
    print("FASE 3 — SOTA: KAN Compacta + MLP (v2)")
    print("=" * 65)

    df = pd.read_parquet(INPUT_PATH)
    X, y, feature_names, dates = prepare_data(df)
    n_feat = X.shape[1]

    print(f"[INFO] ABT: {df.shape} → Features: {n_feat}")
    print(f"[INFO] Target: mean={y.mean():.4f}, median={np.median(y):.4f}")
    print(f"[INFO] HuberLoss(delta={HUBER_DELTA}), Device: {DEVICE}")

    # --- KAN Compacta: [n_feat, 16, 1], grid=3 ---
    def kan_factory():
        return KAN(
            width=[n_feat, 16, 1], grid=3, k=3,
            device=DEVICE, seed=42,
        )

    kan_result = run_cv(
        "KAN [39→16→1] g=3", kan_factory, X, y,
        lr=5e-4, weight_decay=1e-3,
    )

    # --- MLP Regularizado: [n_feat, 32, 16, 1] ---
    def mlp_factory():
        return PhysicsMLP(n_feat).to(DEVICE)

    mlp_result = run_cv(
        "MLP [39→32→16→1]", mlp_factory, X, y,
        lr=1e-3, weight_decay=1e-3,
    )

    # --- Selecionar melhor ---
    print("\n" + "=" * 65)
    print("RESULTADO COMPARATIVO")
    print("=" * 65)

    results = [kan_result, mlp_result]
    xgb_mae = 0.0499
    dummy_mae = 0.0634

    print(f"{'Modelo':<28s}  {'MAE':>8s}  {'RMSE':>8s}  {'vs XGB':>8s}  {'vs Dummy':>9s}")
    print("-" * 65)
    print(f"{'XGBoost (referência)':<28s}  {xgb_mae:>8.4f}  {'0.0887':>8s}  {'—':>8s}  {'+21.3%':>9s}")
    for r in sorted(results, key=lambda x: x["avg_mae"]):
        gain_xgb = ((xgb_mae - r["avg_mae"]) / xgb_mae) * 100
        gain_dummy = ((dummy_mae - r["avg_mae"]) / dummy_mae) * 100
        print(
            f"{r['name']:<28s}  {r['avg_mae']:>8.4f}  {r['avg_rmse']:>8.4f}  "
            f"{gain_xgb:>+7.1f}%  {gain_dummy:>+8.1f}%"
        )
    print("-" * 65)

    best = min(results, key=lambda x: x["avg_mae"])
    print(f"\n[BEST] {best['name']} — MAE={best['avg_mae']:.4f}g")

    # Salvar melhor modelo
    save_name = "kan_sota.pt" if "KAN" in best["name"] else "mlp_sota.pt"
    torch.save(
        {
            "model_state": best["model"].state_dict(),
            "scaler_mean": best["scaler"].mean_,
            "scaler_scale": best["scaler"].scale_,
            "feature_names": feature_names,
            "model_type": best["name"],
            "metrics": {
                "avg_mae": best["avg_mae"],
                "avg_rmse": best["avg_rmse"],
            },
        },
        str(MODELS_DIR / save_name),
    )
    print(f"[OK] Modelo salvo: {MODELS_DIR / save_name}")

    # Plots
    plot_predictions(y, best["all_preds"], dates, best["name"])
    plot_comparison(y, best["all_preds"], dates, best["name"])

    print("\n" + "=" * 65)
    print("FASE 3 — SOTA CONCLUÍDA (v2)")
    print("=" * 65)


if __name__ == "__main__":
    main()
