"""
ETL Pipeline — Predição de Sujidade em Praia (Mineração)
Fase 1: Limpeza, Casting e Construção da ABT (Analytical Base Table)

Fontes de dados:
    - RAMPS: Meteorologia de alta frequência (~15min)
    - POLIGONOS: Emissões calculadas por polígono (~15min)
    - CARREGAMENTO: Dados de processo portuário (horário)
    - TARGET (Y): PESO DO FILTRO diário (coleta ~09:00)

Autor: Jorge Metri / Claude (MLOps)
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ===================================================================
# 1. CLEAN RAMPS
# ===================================================================
def clean_ramps(path: Path | None = None) -> pd.DataFrame:
    """Limpa e transforma dados meteorológicos das RAMPs.

    Regras (TASK.md §2A):
        - Vento: usar sensor 10.0m, fallback para 6.0m
        - Partículas: usar 9.0m e 16.0m (descartar 10.0m)
        - Engenharia vetorial: decompor vento em componentes u, v
        - PROIBIDO: média aritmética de graus
    """
    if path is None:
        path = RAW_DIR / "ramps_2025.csv"

    df = pd.read_csv(path)

    # --- Casting temporal ---
    df["Data e Hora"] = pd.to_datetime(df["Data e Hora"])

    # --- Feature Selection: Vento (fallback 10m -> 6m) ---
    df["direcao_vento"] = df["DirecaoVento [°] 10.0m"].fillna(
        df["DirecaoVento [°] 6.0m"]
    )
    df["velocidade_vento"] = df["VelocidadeVento [m/s] 10.0m"].fillna(
        df["VelocidadeVento [m/s] 6.0m"]
    )

    # --- Partículas: 9m e 16m ---
    df["particulas_9m"] = df["Particulas [µg/m³] 9.0m"]
    df["particulas_16m"] = df["Particulas [µg/m³] 16.0m"]

    # --- Engenharia Vetorial (CRÍTICA) ---
    # Convenção meteorológica: direção DE ONDE o vento vem (0° = Norte, 90° = Leste)
    # u = componente zonal (W->E positivo), v = componente meridional (S->N positivo)
    direcao_rad = np.deg2rad(df["direcao_vento"])
    df["vento_u"] = -df["velocidade_vento"] * np.sin(direcao_rad)
    df["vento_v"] = -df["velocidade_vento"] * np.cos(direcao_rad)

    # --- Selecionar colunas finais ---
    cols_final = [
        "Data e Hora",
        "Origem_Arquivo",
        "direcao_vento",
        "velocidade_vento",
        "vento_u",
        "vento_v",
        "particulas_9m",
        "particulas_16m",
    ]
    df = df[cols_final].copy()
    df = df.rename(columns={"Data e Hora": "datetime", "Origem_Arquivo": "ramp_id"})

    return df


# ===================================================================
# 2. CLEAN POLIGONOS
# ===================================================================
def clean_poligonos(path: Path | None = None) -> pd.DataFrame:
    """Limpa dados de emissão por polígono e pivota para formato wide.

    Regras (TASK.md §2B):
        - Pivotar de Long para Wide (uma coluna por polígono)
        - Se polígono específico é NaN mas outros existem → 0.0
        - Se TODOS os polígonos NaN no timestamp → manter NaN (falha de sistema)
    """
    if path is None:
        path = RAW_DIR / "poligonos_2025.csv"

    df = pd.read_csv(path)

    # --- Strip nomes de coluna (espaço no início) ---
    df.columns = df.columns.str.strip()

    # --- Casting temporal ---
    df["Data e Hora"] = pd.to_datetime(df["Data e Hora"])

    # --- Renomear coluna de emissão ---
    df = df.rename(columns={
        "Data e Hora": "datetime",
        "Particulas [kg/h] 0.0 m": "emissao_kgh",
        "Origem_Arquivo": "origem",
    })

    # --- Mapear nomes de origem para colunas limpas ---
    nome_map = {
        "Torre C3": "emissao_torre_c3",
        "Usina3": "emissao_usina3",
        "Usina 4": "emissao_usina4",
        "Píer de materiais": "emissao_pier",
        "Pátio de estocagem": "emissao_patio",
    }
    df["col_pivot"] = df["origem"].map(nome_map)

    # --- Pivotagem para Wide ---
    df_wide = df.pivot_table(
        index="datetime",
        columns="col_pivot",
        values="emissao_kgh",
        aggfunc="mean",
    ).reset_index()

    df_wide.columns.name = None

    # --- Imputação: NaN parcial → 0.0, NaN total → mantém NaN ---
    emission_cols = [c for c in df_wide.columns if c.startswith("emissao_")]
    all_nan_mask = df_wide[emission_cols].isna().all(axis=1)
    df_wide[emission_cols] = df_wide[emission_cols].fillna(0.0)
    df_wide.loc[all_nan_mask, emission_cols] = np.nan

    return df_wide


# ===================================================================
# 3. CLEAN CARREGAMENTO
# ===================================================================
def clean_carregamento(path: Path | None = None) -> pd.DataFrame:
    """Limpa dados de processo portuário.

    Regras (TASK.md §2C):
        - Header multi-linha: skiprows=7, nomes manuais
        - H2O > 25% → NaN (ruído de sensor)
        - Corrigir vírgula decimal em Finos e Tempo Estoque
        - Flag is_loading
        - Produto NaN → 'NO_OP'
        - H2O/Tempo Estoque NaN → Rolling Median 7 dias
    """
    if path is None:
        path = RAW_DIR / "carregamento_2025.csv"

    col_names = [
        "datetime",
        "carregamento_tmn",
        "navio_atracado",
        "produto",
        "h2o_pct",
        "finos_pct",
        "compressao_16",
        "compressao_12_5",
        "tempo_estoque_dias",
        "descarga_pet_coke",
        "descarga_calcario",
    ]

    df = pd.read_csv(path, header=None, skiprows=7, names=col_names)

    # --- Casting temporal ---
    df["datetime"] = pd.to_datetime(df["datetime"])

    # --- Corrigir strings → float (vírgula decimal ou ponto) ---
    for col in ["finos_pct", "tempo_estoque_dias"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )

    # --- Sanity Check Físico: H2O > 25% é impossível ---
    df.loc[df["h2o_pct"] > 25.0, "h2o_pct"] = np.nan

    # --- Flag binária: is_loading ---
    df["is_loading"] = (df["carregamento_tmn"] > 0).astype(int)

    # --- Produto: NaN → 'NO_OP' ---
    df["produto"] = df["produto"].fillna("NO_OP")

    # --- Imputação Rolling Median 7 dias para H2O e Tempo Estoque ---
    for col in ["h2o_pct", "tempo_estoque_dias"]:
        rolling_med = df[col].rolling(window=7 * 24, min_periods=1, center=True).median()
        monthly_med = df.groupby(df["datetime"].dt.month)[col].transform("median")
        df[col] = df[col].fillna(rolling_med).fillna(monthly_med)

    return df


# ===================================================================
# 4. CLEAN TARGET
# ===================================================================
def clean_target(path: Path | None = None) -> pd.DataFrame:
    """Limpa a variável alvo (PESO DO FILTRO).

    Regras (TASK.md §2D):
        - Remover dias sem PESO DO FILTRO (NaN → sem treino supervisionado)
        - Manter CLASSE DO FILTRO apenas para validação estratificada
    """
    if path is None:
        path = RAW_DIR / "y_2025.csv"

    df = pd.read_csv(path)

    # --- Casting temporal ---
    df["Data"] = pd.to_datetime(df["Data"])

    # --- Renomear ---
    df = df.rename(columns={
        "Data": "data",
        "PESO DO FILTRO": "peso_filtro",
        "CLASSE DO FILTRO": "classe_filtro",
    })

    # --- Corrigir vírgula decimal se necessário ---
    if df["peso_filtro"].dtype == object:
        df["peso_filtro"] = pd.to_numeric(
            df["peso_filtro"].astype(str).str.replace(",", "."), errors="coerce"
        )

    # --- Dropar dias sem target ---
    n_antes = len(df)
    df = df.dropna(subset=["peso_filtro"]).reset_index(drop=True)
    n_depois = len(df)
    print(f"[TARGET] Removidos {n_antes - n_depois} dias sem PESO DO FILTRO "
          f"({n_depois} dias válidos restantes)")

    return df


# ===================================================================
# 5. CREATE ABT (Analytical Base Table)
# ===================================================================
def create_abt(
    df_target: pd.DataFrame,
    df_ramps: pd.DataFrame,
    df_poly: pd.DataFrame,
    df_carreg: pd.DataFrame,
) -> pd.DataFrame:
    """Cria a ABT diária unificando todas as fontes.

    Regras (TASK.md §3):
        - Para cada data D válida do target (coleta ~09:00):
          Janela de integração: [D-1 09:00, D 09:00]
        - RAMPS: média vetorial u,v; max de partículas
        - POLIGONOS: integral de Riemann (massa total emitida)
        - CARREGAMENTO: soma toneladas, moda produto, média H2O
    """
    HORA_COLETA = 9
    DELTA_T_RAMPS_H = 0.25  # 15 minutos em horas
    DELTA_T_POLY_H = 0.25

    emission_cols = [c for c in df_poly.columns if c.startswith("emissao_")]
    records = []

    for _, row in df_target.iterrows():
        d = row["data"]
        t_fim = pd.Timestamp(d.year, d.month, d.day, HORA_COLETA)
        t_ini = t_fim - pd.Timedelta(hours=24)

        # --- RAMPS: janela [D-1 09:00, D 09:00] ---
        mask_r = (df_ramps["datetime"] >= t_ini) & (df_ramps["datetime"] < t_fim)
        ramps_win = df_ramps.loc[mask_r]

        u_mean = ramps_win["vento_u"].mean()
        v_mean = ramps_win["vento_v"].mean()
        vel_mean = ramps_win["velocidade_vento"].mean()
        vel_max = ramps_win["velocidade_vento"].max()
        part_9m_max = ramps_win["particulas_9m"].max()
        part_16m_max = ramps_win["particulas_16m"].max()
        part_9m_mean = ramps_win["particulas_9m"].mean()
        part_16m_mean = ramps_win["particulas_16m"].mean()

        # Direção média reconstruída a partir dos vetores
        dir_media = np.degrees(np.arctan2(-u_mean, -v_mean)) % 360

        # --- POLIGONOS: integral de Riemann (kg/h * Δt_h = kg) ---
        mask_p = (df_poly["datetime"] >= t_ini) & (df_poly["datetime"] < t_fim)
        poly_win = df_poly.loc[mask_p]

        poly_feats = {}
        for col in emission_cols:
            massa_col = col.replace("emissao_", "massa_")
            poly_feats[massa_col] = poly_win[col].sum() * DELTA_T_POLY_H

        emissao_total = sum(poly_feats.values())

        # --- CARREGAMENTO: agregações na janela ---
        mask_c = (df_carreg["datetime"] >= t_ini) & (df_carreg["datetime"] < t_fim)
        carreg_win = df_carreg.loc[mask_c]

        carreg_total = carreg_win["carregamento_tmn"].sum()
        is_loading_any = int(carreg_win["is_loading"].any())
        h2o_media = carreg_win["h2o_pct"].mean()
        tempo_estoque_media = carreg_win["tempo_estoque_dias"].mean()
        descarga_pet_coke = carreg_win["descarga_pet_coke"].sum()
        descarga_calcario = carreg_win["descarga_calcario"].sum()

        # Moda do produto
        produtos = carreg_win.loc[
            carreg_win["produto"] != "NO_OP", "produto"
        ]
        produto_moda = produtos.mode().iloc[0] if len(produtos) > 0 else "NO_OP"

        # --- Montar registro ---
        record = {
            "data": d,
            "peso_filtro": row["peso_filtro"],
            "classe_filtro": row["classe_filtro"],
            # Vento
            "vento_u_mean": u_mean,
            "vento_v_mean": v_mean,
            "vento_vel_mean": vel_mean,
            "vento_vel_max": vel_max,
            "vento_dir_media": dir_media,
            # Partículas
            "part_9m_max": part_9m_max,
            "part_16m_max": part_16m_max,
            "part_9m_mean": part_9m_mean,
            "part_16m_mean": part_16m_mean,
            # Emissões (massa total em kg)
            **poly_feats,
            "emissao_total_kg": emissao_total,
            # Carregamento
            "carreg_total_tmn": carreg_total,
            "is_loading": is_loading_any,
            "produto_moda": produto_moda,
            "h2o_media": h2o_media,
            "tempo_estoque_media": tempo_estoque_media,
            "descarga_pet_coke": descarga_pet_coke,
            "descarga_calcario": descarga_calcario,
        }
        records.append(record)

    abt = pd.DataFrame(records)

    print(f"[ABT] Shape final: {abt.shape}")
    print(f"[ABT] Colunas: {list(abt.columns)}")
    print(f"[ABT] NaN por coluna:\n{abt.isna().sum()}")

    return abt


# ===================================================================
# MAIN — Orquestrador
# ===================================================================
def main():
    print("=" * 60)
    print("ETL Pipeline — Fase 1")
    print("=" * 60)

    # --- 1. Limpar cada fonte ---
    print("\n[1/5] Limpando RAMPS...")
    df_ramps = clean_ramps()
    print(f"       Shape: {df_ramps.shape}")

    print("\n[2/5] Limpando POLIGONOS...")
    df_poly = clean_poligonos()
    print(f"       Shape: {df_poly.shape}")

    print("\n[3/5] Limpando CARREGAMENTO...")
    df_carreg = clean_carregamento()
    print(f"       Shape: {df_carreg.shape}")

    print("\n[4/5] Limpando TARGET...")
    df_target = clean_target()
    print(f"       Shape: {df_target.shape}")

    # --- 2. Construir ABT ---
    print("\n[5/5] Construindo ABT...")
    abt = create_abt(df_target, df_ramps, df_poly, df_carreg)

    # --- 3. Salvar ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "abt_master.parquet"
    abt.to_parquet(out_path, index=False)
    print(f"\n[OK] ABT salva em: {out_path}")
    print(f"[OK] Shape: {abt.shape}")

    return abt


if __name__ == "__main__":
    main()
