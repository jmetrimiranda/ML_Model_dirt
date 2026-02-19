"""
EDA — Análise Exploratória dos Dados (Fase 2)
Gera gráficos de alta qualidade para validação física e estatística.

Autor: Jorge Metri / Claude (MLOps)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config global
# ---------------------------------------------------------------------------
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

sns.set_theme(style="whitegrid", palette="muted")


def load_abt() -> pd.DataFrame:
    abt = pd.read_parquet("data/processed/abt_master.parquet")
    abt["data"] = pd.to_datetime(abt["data"])
    return abt


# ===================================================================
# PASSO 1: Análise Univariada & Temporal
# ===================================================================
def passo1_temporal(abt: pd.DataFrame):
    """TASK.md §4 Passo 1: Line Chart + Histograma + KDE."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

    # --- 1A. Evolução Temporal ---
    ax = axes[0]
    ax.plot(abt["data"], abt["peso_filtro"], linewidth=0.8, alpha=0.8, color="#2196F3")
    ax.fill_between(abt["data"], abt["peso_filtro"], alpha=0.15, color="#2196F3")

    # Média móvel 7 dias
    rolling_7d = abt.set_index("data")["peso_filtro"].rolling("7D").mean()
    ax.plot(rolling_7d.index, rolling_7d.values, linewidth=2, color="#F44336",
            label="Média Móvel 7d", linestyle="--")

    ax.set_title("Evolução Temporal — PESO DO FILTRO (g)")
    ax.set_ylabel("Peso (g)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    # --- 1B. Histograma + KDE ---
    ax2 = axes[1]
    peso = abt["peso_filtro"].dropna()
    ax2.hist(peso, bins=40, density=True, alpha=0.6, color="#2196F3", edgecolor="white")
    peso_log = peso[peso > 0]

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(peso)
    x_range = np.linspace(peso.min(), peso.max(), 200)
    ax2.plot(x_range, kde(x_range), linewidth=2, color="#F44336", label="KDE")

    # Teste Log-Normalidade (Shapiro em log(y) para y > 0)
    if len(peso_log) > 3:
        stat_sw, p_sw = stats.shapiro(np.log(peso_log))
        ax2.text(0.97, 0.85,
                 f"Shapiro-Wilk (log y>0)\nW={stat_sw:.4f}, p={p_sw:.4f}\n"
                 f"{'Log-Normal' if p_sw > 0.05 else 'NÃO Log-Normal'}",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax2.set_title("Distribuição — PESO DO FILTRO")
    ax2.set_xlabel("Peso (g)")
    ax2.set_ylabel("Densidade")
    ax2.legend()

    plt.tight_layout()
    path = FIGURES_DIR / "01_evolucao_temporal_peso.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[EDA 1] Salvo: {path}")


# ===================================================================
# PASSO 2: Análise Bivariada Operacional (Teste A/B)
# ===================================================================
def passo2_bivariada(abt: pd.DataFrame):
    """TASK.md §4 Passo 2: Box Plot + Mann-Whitney U Test."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- 2A. Box Plot Condicional ---
    ax = axes[0]
    grupos = {
        "Sem Navio\n(is_loading=0)": abt[abt["is_loading"] == 0]["peso_filtro"],
        "Com Navio\n(is_loading=1)": abt[abt["is_loading"] == 1]["peso_filtro"],
    }
    bp_data = list(grupos.values())
    bp_labels = list(grupos.keys())

    bp = ax.boxplot(bp_data, tick_labels=bp_labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#64B5F6", "#EF5350"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Peso do Filtro por Estado Operacional")
    ax.set_ylabel("Peso (g)")

    # Estatísticas descritivas no gráfico
    for i, (label, data) in enumerate(grupos.items()):
        ax.text(i + 1, data.max() + 0.02,
                f"n={len(data)}\nmed={data.median():.3f}\nmu={data.mean():.3f}",
                ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # --- 2B. Violin Plot + Mann-Whitney ---
    ax2 = axes[1]
    plot_df = abt[["peso_filtro", "is_loading"]].copy()
    plot_df["Operação"] = plot_df["is_loading"].map({0: "Sem Navio", 1: "Com Navio"})

    sns.violinplot(data=plot_df, x="Operação", y="peso_filtro", hue="Operação",
                   palette=["#64B5F6", "#EF5350"], inner="quartile", cut=0,
                   ax=ax2, legend=False)
    sns.stripplot(data=plot_df, x="Operação", y="peso_filtro", ax=ax2,
                  color="black", alpha=0.2, size=3, jitter=True)

    # Mann-Whitney U Test
    g0 = abt[abt["is_loading"] == 0]["peso_filtro"]
    g1 = abt[abt["is_loading"] == 1]["peso_filtro"]
    u_stat, p_val = stats.mannwhitneyu(g0, g1, alternative="two-sided")

    sig = "SIGNIFICANTE (p < 0.05)" if p_val < 0.05 else "NÃO significante"
    ax2.set_title(f"Mann-Whitney U Test\nU={u_stat:.0f}, p={p_val:.4f} — {sig}")
    ax2.set_ylabel("Peso (g)")

    plt.tight_layout()
    path = FIGURES_DIR / "02_bivariada_operacional.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[EDA 2] Salvo: {path}")

    # Print do resultado
    print(f"\n  --- Mann-Whitney U Test ---")
    print(f"  Sem Navio (n={len(g0)}): mediana={g0.median():.4f}, média={g0.mean():.4f}")
    print(f"  Com Navio (n={len(g1)}): mediana={g1.median():.4f}, média={g1.mean():.4f}")
    print(f"  U={u_stat:.0f}, p-value={p_val:.6f}")
    print(f"  Conclusão: {sig}")


# ===================================================================
# PASSO 3: Detecção de Anomalias (Chebyshev)
# ===================================================================
def passo3_chebyshev(abt: pd.DataFrame):
    """TASK.md §4 Passo 3: Chebyshev k=3, cruzar outliers com vento."""

    peso = abt["peso_filtro"]
    mu = peso.mean()
    sigma = peso.std()

    for k in [3, 4]:
        limiar = mu + k * sigma
        outliers = abt[peso > limiar].copy()
        print(f"\n  --- Chebyshev k={k} (limiar={limiar:.4f}g) ---")
        print(f"  Outliers encontrados: {len(outliers)} / {len(abt)}")
        if len(outliers) > 0:
            cols_show = ["data", "peso_filtro", "vento_vel_max", "vento_dir_media",
                         "fluxo_efetivo", "emissao_total_kg", "is_loading",
                         "precipitacao_mm"]
            print(outliers[cols_show].to_string(index=False))

    # Gráfico: timeline com outliers marcados
    k = 3
    limiar = mu + k * sigma
    outliers = abt[peso > limiar]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(abt["data"], abt["peso_filtro"], linewidth=0.8, alpha=0.6, color="#2196F3")
    ax.axhline(y=mu, color="green", linestyle="--", linewidth=1, label=f"Média ({mu:.3f}g)")
    ax.axhline(y=limiar, color="red", linestyle="--", linewidth=1.5,
               label=f"Chebyshev k=3 ({limiar:.3f}g)")
    ax.scatter(outliers["data"], outliers["peso_filtro"],
               c=outliers["vento_vel_max"], cmap="YlOrRd", s=80, zorder=5,
               edgecolors="black", linewidth=0.8)
    cbar = plt.colorbar(ax.collections[-1], ax=ax, shrink=0.7)
    cbar.set_label("Vel. Vento Máx (m/s)")

    ax.set_title(f"Outliers Chebyshev (k=3): {len(outliers)} eventos extremos")
    ax.set_ylabel("Peso (g)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    path = FIGURES_DIR / "03_chebyshev_outliers.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"\n[EDA 3] Salvo: {path}")


# ===================================================================
# PASSO 4: Windrose + Cross-Correlation
# ===================================================================
def passo4_fisica(abt: pd.DataFrame):
    """TASK.md §4 Passo 4: Windrose ponderada + CCF(Fluxo, Peso)."""

    # --- 4A. Rosa dos Ventos ponderada pelo peso ---
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    dir_rad = np.deg2rad(abt["vento_dir_media"])
    peso = abt["peso_filtro"]
    vel = abt["vento_vel_mean"]

    # Binning por direção (16 setores de 22.5°)
    n_sectors = 16
    sector_width = 2 * np.pi / n_sectors
    sector_edges = np.linspace(0, 2 * np.pi, n_sectors + 1)

    sector_peso_mean = []
    sector_vel_mean = []
    sector_counts = []
    for i in range(n_sectors):
        mask = (dir_rad >= sector_edges[i]) & (dir_rad < sector_edges[i + 1])
        sector_peso_mean.append(peso[mask].mean() if mask.sum() > 0 else 0)
        sector_vel_mean.append(vel[mask].mean() if mask.sum() > 0 else 0)
        sector_counts.append(mask.sum())

    sector_centers = (sector_edges[:-1] + sector_edges[1:]) / 2
    peso_arr = np.array(sector_peso_mean)
    vel_arr = np.array(sector_vel_mean)

    # Normalizar para visualização
    colors = cm.YlOrRd(peso_arr / peso_arr.max() if peso_arr.max() > 0 else peso_arr)
    bars = ax.bar(sector_centers, vel_arr, width=sector_width * 0.85,
                  color=colors, edgecolor="gray", linewidth=0.5, alpha=0.85)

    # Anotações
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Rosa dos Ventos\n(altura = vel. média, cor = peso médio)", pad=20)

    # Colorbar
    sm = cm.ScalarMappable(cmap="YlOrRd",
                           norm=plt.Normalize(peso_arr.min(), peso_arr.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Peso Médio (g)")

    path = FIGURES_DIR / "03_windrose_poluicao.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[EDA 4A] Salvo: {path}")

    # --- 4B. Cross-Correlation: Fluxo_Efetivo vs Peso ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fluxo = abt["fluxo_efetivo"].values
    peso_v = abt["peso_filtro"].values
    max_lag = 5

    # CCF manual (correlação cruzada normalizada)
    lags = range(-max_lag, max_lag + 1)
    ccf_values = []
    for lag in lags:
        if lag >= 0:
            x = fluxo[:len(fluxo) - lag]
            y = peso_v[lag:]
        else:
            x = fluxo[-lag:]
            y = peso_v[:len(peso_v) + lag]
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() > 2:
            corr, _ = stats.pearsonr(x[valid], y[valid])
        else:
            corr = 0
        ccf_values.append(corr)

    ax1 = axes[0]
    bars = ax1.bar(list(lags), ccf_values, color="#2196F3", edgecolor="white", alpha=0.8)
    # Destacar lag=0
    best_lag = list(lags)[np.argmax(np.abs(ccf_values))]
    for bar, lag in zip(bars, lags):
        if lag == best_lag:
            bar.set_color("#F44336")

    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_title("Cross-Correlation: Fluxo Efetivo vs Peso")
    ax1.set_xlabel("Lag (dias)")
    ax1.set_ylabel("Pearson r")
    ax1.set_xticks(list(lags))

    # --- Scatter: Fluxo vs Peso ---
    ax2 = axes[1]
    valid = ~(np.isnan(fluxo) | np.isnan(peso_v))
    ax2.scatter(fluxo[valid], peso_v[valid], alpha=0.4, s=20, c="#2196F3", edgecolors="none")
    r, p = stats.pearsonr(fluxo[valid], peso_v[valid])

    # Regressão linear para tendência
    z = np.polyfit(fluxo[valid], peso_v[valid], 1)
    x_line = np.linspace(fluxo[valid].min(), fluxo[valid].max(), 100)
    ax2.plot(x_line, np.polyval(z, x_line), color="#F44336", linewidth=2, linestyle="--")

    ax2.set_title(f"Fluxo Efetivo vs Peso\nr={r:.3f}, p={p:.4f}")
    ax2.set_xlabel("Fluxo Efetivo (m/s)")
    ax2.set_ylabel("Peso (g)")

    plt.tight_layout()
    path = FIGURES_DIR / "04_cross_correlation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[EDA 4B] Salvo: {path}")
    print(f"  Pearson(fluxo, peso) lag=0: r={r:.4f}, p={p:.6f}")
    print(f"  Melhor lag: {best_lag} dias (r={ccf_values[list(lags).index(best_lag)]:.4f})")


# ===================================================================
# PASSO 5: PCA de Partículas
# ===================================================================
def passo5_pca(abt: pd.DataFrame):
    """TASK.md §4 Passo 5: PCA para verificar redundância entre sensores."""

    particle_cols = ["part_9m_max", "part_16m_max", "part_9m_mean", "part_16m_mean"]
    X = abt[particle_cols].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Scree Plot ---
    ax = axes[0]
    pcs = range(1, len(explained) + 1)
    ax.bar(pcs, explained * 100, color="#2196F3", edgecolor="white", alpha=0.8)
    ax.plot(pcs, cumulative * 100, "o-", color="#F44336", linewidth=2)
    ax.axhline(y=90, color="green", linestyle="--", linewidth=1, label="90% limiar")

    for i, (exp, cum) in enumerate(zip(explained, cumulative)):
        ax.text(i + 1, exp * 100 + 1.5, f"{exp * 100:.1f}%", ha="center", fontsize=9)

    ax.set_title("Scree Plot — PCA de Partículas")
    ax.set_xlabel("Componente Principal")
    ax.set_ylabel("Variância Explicada (%)")
    ax.set_xticks(list(pcs))
    ax.legend()

    # --- Loadings (PC1 e PC2) ---
    ax2 = axes[1]
    loadings = pd.DataFrame(
        pca.components_[:2].T,
        columns=["PC1", "PC2"],
        index=particle_cols,
    )
    loadings.plot(kind="barh", ax=ax2, color=["#2196F3", "#EF5350"], alpha=0.8)
    ax2.set_title("Loadings — PC1 e PC2")
    ax2.set_xlabel("Peso no Componente")
    ax2.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = FIGURES_DIR / "05_pca_scree_plot.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[EDA 5] Salvo: {path}")
    print(f"  PC1 explica {explained[0] * 100:.1f}% da variância")
    print(f"  PC1+PC2 explica {cumulative[1] * 100:.1f}% da variância")
    if cumulative[0] >= 0.9:
        print(f"  CONCLUSÃO: PC1 > 90% → pode usar PC1 como proxy único de poluição")
    else:
        print(f"  CONCLUSÃO: PC1 < 90% → manter features separadas ou usar PC1+PC2")


# ===================================================================
# PASSO 6: Washout (Efeito da Chuva)
# ===================================================================
def passo6_washout(abt: pd.DataFrame):
    """TASK.md §4 Passo 6: Scatter Fluxo vs Peso colorido por chuva + Mann-Whitney."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- 6A. Scatter colorido ---
    ax = axes[0]
    dry = abt[abt["is_rainy"] == 0]
    wet = abt[abt["is_rainy"] == 1]

    ax.scatter(dry["fluxo_efetivo"], dry["peso_filtro"],
               alpha=0.5, s=30, c="#FF9800", label=f"Seco (n={len(dry)})",
               edgecolors="none")
    ax.scatter(wet["fluxo_efetivo"], wet["peso_filtro"],
               alpha=0.5, s=30, c="#2196F3", label=f"Chuva (n={len(wet)})",
               edgecolors="none", marker="^")

    ax.set_title("Fluxo Efetivo vs Peso — Efeito da Chuva")
    ax.set_xlabel("Fluxo Efetivo (m/s)")
    ax.set_ylabel("Peso (g)")
    ax.legend()

    # --- 6B. Box Plot condicional: Alta emissão seca vs úmida ---
    ax2 = axes[1]

    # Definir "alta emissão" como acima da mediana do fluxo_efetivo
    high_flux_thresh = abt["fluxo_efetivo"].median()
    high_flux = abt[abt["fluxo_efetivo"] > high_flux_thresh]

    hf_dry = high_flux[high_flux["is_rainy"] == 0]["peso_filtro"]
    hf_wet = high_flux[high_flux["is_rainy"] == 1]["peso_filtro"]

    bp_data = [hf_dry, hf_wet]
    bp_labels = [f"Alta Emissão\n+ Seco\n(n={len(hf_dry)})",
                 f"Alta Emissão\n+ Chuva\n(n={len(hf_wet)})"]

    bp = ax2.boxplot(bp_data, tick_labels=bp_labels, patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#FF9800")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#2196F3")
    bp["boxes"][1].set_alpha(0.7)

    # Mann-Whitney
    if len(hf_dry) > 5 and len(hf_wet) > 5:
        u_stat, p_val = stats.mannwhitneyu(hf_dry, hf_wet, alternative="two-sided")
        sig = "SIGNIFICANTE" if p_val < 0.05 else "NÃO significante"
        ax2.set_title(f"Washout Test (Alta Emissão)\nMann-Whitney U={u_stat:.0f}, p={p_val:.4f} — {sig}")
        print(f"\n  --- Washout Test (Alta Emissão: fluxo > {high_flux_thresh:.3f}) ---")
        print(f"  Seco (n={len(hf_dry)}): mediana={hf_dry.median():.4f}, média={hf_dry.mean():.4f}")
        print(f"  Chuva (n={len(hf_wet)}): mediana={hf_wet.median():.4f}, média={hf_wet.mean():.4f}")
        print(f"  U={u_stat:.0f}, p={p_val:.6f} → {sig}")
        if p_val < 0.05:
            print(f"  CONCLUSÃO: Chuva é regressor negativo OBRIGATÓRIO no modelo")
        else:
            print(f"  CONCLUSÃO: Efeito washout não significante neste recorte")
    else:
        ax2.set_title("Washout Test — amostras insuficientes")

    ax2.set_ylabel("Peso (g)")

    plt.tight_layout()
    path = FIGURES_DIR / "06_efeito_chuva.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[EDA 6] Salvo: {path}")


# ===================================================================
# PASSO 7: Sensibilidade da Janela de Coleta (08:00-09:00)
# ===================================================================
def passo7_incerteza(abt: pd.DataFrame):
    """TASK.md §4 Passo 7: StdDev do vento na janela 08-09h vs Peso."""

    # Carregar ramps brutos para calcular variância na janela 08-09
    from src.data.make_dataset import clean_ramps
    df_ramps = clean_ramps()

    records = []
    for _, row in abt.iterrows():
        d = row["data"]
        t_08 = pd.Timestamp(d.year, d.month, d.day, 8)
        t_09 = pd.Timestamp(d.year, d.month, d.day, 9)

        mask = (df_ramps["datetime"] >= t_08) & (df_ramps["datetime"] < t_09)
        win = df_ramps.loc[mask]

        records.append({
            "data": d,
            "peso_filtro": row["peso_filtro"],
            "std_vel_0809": win["velocidade_vento"].std(),
            "std_dir_0809": win["direcao_vento"].std(),
            "n_obs_0809": len(win),
        })

    df_incerteza = pd.DataFrame(records).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 7A. StdDev Velocidade vs Peso ---
    ax = axes[0]
    ax.scatter(df_incerteza["std_vel_0809"], df_incerteza["peso_filtro"],
               alpha=0.4, s=20, c="#673AB7", edgecolors="none")

    r1, p1 = stats.pearsonr(
        df_incerteza["std_vel_0809"].values,
        df_incerteza["peso_filtro"].values
    )
    ax.set_title(f"Incerteza Vel. Vento (08-09h) vs Peso\nr={r1:.3f}, p={p1:.4f}")
    ax.set_xlabel("StdDev Velocidade Vento (m/s)")
    ax.set_ylabel("Peso (g)")

    # --- 7B. StdDev Direção vs Peso ---
    ax2 = axes[1]
    ax2.scatter(df_incerteza["std_dir_0809"], df_incerteza["peso_filtro"],
                alpha=0.4, s=20, c="#009688", edgecolors="none")

    r2, p2 = stats.pearsonr(
        df_incerteza["std_dir_0809"].values,
        df_incerteza["peso_filtro"].values
    )
    ax2.set_title(f"Incerteza Dir. Vento (08-09h) vs Peso\nr={r2:.3f}, p={p2:.4f}")
    ax2.set_xlabel("StdDev Direção Vento (°)")
    ax2.set_ylabel("Peso (g)")

    plt.tight_layout()
    path = FIGURES_DIR / "07_incerteza_coleta.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[EDA 7] Salvo: {path}")
    print(f"  Corr(StdVel_0809, Peso): r={r1:.4f}, p={p1:.6f}")
    print(f"  Corr(StdDir_0809, Peso): r={r2:.4f}, p={p2:.6f}")
    if p1 < 0.05 or p2 < 0.05:
        print("  CONCLUSÃO: Volatilidade na hora de coleta correlaciona com peso")
        print("  → Justifica modelos probabilísticos (intervalo de confiança)")
    else:
        print("  CONCLUSÃO: Incerteza da coleta NÃO correlaciona significativamente")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("EDA — Fase 2 (Passos 1-7)")
    print("=" * 60)

    abt = load_abt()
    print(f"ABT carregada: {abt.shape}")

    print("\n[Passo 1] Análise Temporal...")
    passo1_temporal(abt)

    print("\n[Passo 2] Análise Bivariada Operacional...")
    passo2_bivariada(abt)

    print("\n[Passo 3] Detecção de Anomalias (Chebyshev)...")
    passo3_chebyshev(abt)

    print("\n[Passo 4] Análise Física (Windrose + CCF)...")
    passo4_fisica(abt)

    print("\n[Passo 5] PCA de Partículas...")
    passo5_pca(abt)

    print("\n[Passo 6] Washout (Efeito da Chuva)...")
    passo6_washout(abt)

    print("\n[Passo 7] Incerteza da Coleta (08-09h)...")
    passo7_incerteza(abt)

    print("\n" + "=" * 60)
    print("EDA — TODOS OS PASSOS COMPLETOS!")
    print("=" * 60)


if __name__ == "__main__":
    main()