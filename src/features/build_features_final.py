"""
Feature Engineering Final — Fase 3, Passo A
Gera lags físicos e features de interação para modelagem.

Input:  data/processed/abt_master.parquet
Output: data/processed/abt_modeling.parquet

Autor: Jorge Metri / Claude (MLOps)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
INPUT_PATH = PROCESSED_DIR / "abt_master.parquet"
OUTPUT_PATH = PROCESSED_DIR / "abt_modeling.parquet"


def build_modeling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona lags físicos e features de interação à ABT.

    Regras (TASK.md §5.A):
      - fluxo_efetivo_lag1, lag2: deslocamento temporal do fluxo
      - precipitacao_lag1: chuva do dia anterior
      - interacao_vento_chuva: fluxo_efetivo / (precipitacao_mm + 1)
    """
    df = df.sort_values("data").reset_index(drop=True)

    # --- Lags Físicos (shift opera sobre séries ordenadas por data) ---
    df["fluxo_efetivo_lag1"] = df["fluxo_efetivo"].shift(1)
    df["fluxo_efetivo_lag2"] = df["fluxo_efetivo"].shift(2)
    df["precipitacao_lag1"] = df["precipitacao_mm"].shift(1)

    # --- Interaction Features ---
    df["interacao_vento_chuva"] = df["fluxo_efetivo"] / (df["precipitacao_mm"] + 1)

    # --- Dropar linhas sem lag (primeiras 2 linhas) ---
    df = df.dropna(subset=["fluxo_efetivo_lag2"]).reset_index(drop=True)

    return df


def main() -> None:
    print(f"[INFO] Lendo ABT de {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"[INFO] Shape original: {df.shape}")

    df = build_modeling_features(df)
    print(f"[INFO] Shape após feature engineering: {df.shape}")

    # Resumo das novas features
    new_cols = [
        "fluxo_efetivo_lag1",
        "fluxo_efetivo_lag2",
        "precipitacao_lag1",
        "interacao_vento_chuva",
    ]
    print("\n[INFO] Novas features:")
    print(df[new_cols].describe().round(4).to_string())

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n[OK] ABT de modelagem salva em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()