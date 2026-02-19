# Contexto do Projeto: Predição de Sujidade em Praia (Mineração) - SOTA Architecture

## 0. Infraestrutura & Workflow (DevContainer + DVC + GitFlow)

**ATENÇÃO CLAUDE:** Este projeto roda em um ambiente estrito de MLOps. Antes de executar qualquer código ou sugerir alterações, você deve entender o estado atual do repositório.

### A. Estrutura de Arquivos (Project Tree)
```text
projeto-ml-sota/
│
├── .devcontainer/       # Config do Docker (Python 3.11 + GPU)
├── .dvc/                # Config interna do DVC (não mexer manualmente)
├── .github/             # Pipelines CI/CD
│
├── conf/                # HYDRA: Gestão de Configuração
│   ├── base/
│   │   ├── train.yaml   # Hiperparâmetros
│   │   └── data.yaml    # Caminhos dos dados
│   └── config.yaml      # Entry point do Hydra
│
├── data/                # DVC CONTROLLED (Ignorado pelo Git)
│   ├── raw/             # CSVs Originais (Adicionados manualmente pelo usuário)
│   ├── processed/       # ABT e Parquets gerados pelos scripts
│   └── .gitignore       # Gerado pelo DVC
│
├── models/              # DVC CONTROLLED: Artefatos (.pkl, .pt)
│
├── notebooks/           # Jupyter Notebooks para EDA (Git)
│
├── reports/             # Artefatos de Comunicação
│   ├── figures/         # PNGs/PDFs (High DPI)
│   └── paper/           # LaTeX/Beamer
│       └── main.tex
│
├── src/                 # Código Fonte Python (Git)
│   ├── __init__.py
│   ├── data/            # Scripts de ETL (make_dataset.py)
│   ├── features/        # Feature Engineering
│   ├── models/          # Treinamento e Inferência
│   └── visualization/   # Plots para o paper
│
├── dvc.yaml             # DVC Pipeline (DAG)
├── dvc.lock             # Estado congelado dos dados
├── poetry.lock          # Dependências Python (Lockfile)
├── pyproject.toml       # Configuração do Poetry
└── README.md
```

### B. Regras de Versionamento (GitFlow & DVC)
1.  **Estado Atual:**
    * Estamos na branch `develop` (ou feature branch).
    * Último Commit: `chore: Configuração/Build Inicial` (Autor: Jorge Metri).
    * **Dados Raw:** O usuário irá adicionar os arquivos CSV brutos em `data/raw/` manualmente. Não tente baixar ou criar dados dummy a menos que solicitado.
2.  **Fluxo de Trabalho Obrigatório:**
    * **Código (`src/`, `conf/`):** Versionado pelo **Git**.
    * **Dados (`data/`) e Modelos (`models/`):** Versionados pelo **DVC**.
    * **Dependências:** Gerenciadas via **Poetry**. Nunca use `pip install` direto; use `poetry add`.
3.  **Comandos Permitidos:**
    * Ao gerar um novo dataset processado: `dvc add data/processed/abt_master.parquet`.
    * Ao criar scripts: `git add src/...`.
    * Para rodar pipelines: `dvc repro`.

---

## 1. Objetivo de Negócio
Prever a variável contínua `PESO DO FILTRO` (acumulado 24h) integrando dados meteorológicos de alta frequência (RAMPs), emissões calculadas de fluxo (Polígonos) e dados de processo portuário (Carregamento).
O modelo deve distinguir com precisão entre poeira gerada por operação ativa (Navio) vs. erosão eólica passiva (Pátio), lidando robustamente com a esparsidade dos dados operacionais e incertezas de coleta.

---

## 2. Estratégia de ETL (Data Cleaning & Casting)

O Claude deve criar scripts modulares em `src/data` para tratar cada fonte separadamente antes da unificação.

### A. Tabela `RAMPS` (Meteorologia)
* **Arquivo:** `df_ramps`
* **Diagnóstico:** Dados de alta frequência com falhas intermitentes.
* **Ações Obrigatórias:**
    1.  **Casting de Tempo:** Converter `Data e Hora` para `datetime64[ns]`.
    2.  **Feature Selection:**
        * **Vento:** Usar estritamente `DirecaoVento [°] 10.0m` e `VelocidadeVento [m/s] 10.0m` (Sensor padrão ouro). Se `10.0m` for NaN, tentar imputar com `6.0m`, caso contrário, marcar como `NaN`.
        * **Partículas:** Descartar a coluna `10.0m` (muitos nulos). **Usar `Particulas [µg/m³] 9.0m` e `16.0m`** como as variáveis principais.
    3.  **Engenharia Vetorial (CRÍTICA):**
        * Converter graus para radianos.
        * Calcular componentes $u$ (zonal) e $v$ (meridional) *imediatamente* após o carregamento.
        * **Proibido:** Fazer média aritmética de graus.

### B. Tabela `POLIGONOS` (Emissões)
* **Arquivo:** `df_poly`
* **Diagnóstico:** Coluna `Particulas [kg/h]` contém números formatados como texto (ex: "1,09").
* **Ações Obrigatórias:**
    1.  **String Cleaning:**
        * Detectar se a coluna é `object`. Se sim: `str.replace(',', '.')` e converter para `float`.
    2.  **Pivotagem (Wide Format):**
        * Transformar de Long (`Data`, `Origem`, `Valor`) para Wide (`Data`, `Emissao_TorreC3`, `Emissao_Patio`, ...).
    3.  **Imputação de Emissão:**
        * Se um polígono específico não tem dados num timestamp, preencher com `0.0` (assumir ausência de emissão calculada).
        * Manter `NaN` apenas se *todos* os polígonos estiverem vazios no timestamp (falha de sistema).

### C. Tabela `CARREGAMENTO` (Processo - Handling Complexo)
* **Arquivo:** `df_proc`
* **Diagnóstico:**
    * Muitos zeros (dias sem navio).
    * `H20 (%)` com outliers absurdos (>100%).
    * Colunas `Finos` e `Tempo Estoque` formatadas com vírgula (`object`).
* **Ações Obrigatórias:**
    1.  **Sanity Check (Físico):**
        * `H20 (%)`: Valores > 25.0 são ruído de sensor. Substituir por `NaN`.
    2.  **Casting:**
        * Corrigir vírgulas em `Finos` e `Tempo Estoque`.
    3.  **Estratégia "Carregamento Zero" (SOTA Imputation):**
        * Quando `Carregamento == 0`, as colunas `H2O`, `Produto`, `Tempo Estoque` são nulas.
        * **NÃO PREENCHER COM ZERO.** Isso criaria um viés de "minério seco".
        * **Ação 1 (Flag):** Criar feature binária `is_loading` (1 se Carregamento > 0, else 0).
        * **Ação 2 (Produto):** Preencher nulos com a categoria string `'NO_OP'`.
        * **Ação 3 (H2O e Estoque):** Preencher nulos usando **Rolling Median (janela 7 dias)** ou Mediana Global do Mês. Isso representa o "estado basal" das pilhas de minério no pátio quando não há navio.

### D. Tabela `TARGET` (Y)
* **Arquivo:** `df_y`
* **Ação:** Remover dias onde `PESO DO FILTRO` é `NaN`. Sem target, sem treino supervisionado.
* **Manter:** `CLASSE DO FILTRO` apenas para validação estratificada, não como feature de entrada.

---

## 3. Estratégia de Unificação (The Master Table)

**Função:** `create_abt(df_y, df_ramps, df_poly, df_carreg)`

Iterar sobre cada `Data` válida do Target ($D$, coleta às 09:00):
1.  **Janela de Integração:** Intervalo $[D_{-1} \text{ 09:00}, D \text{ 09:00}]$.
2.  **Agregações:**
    * **RAMPS:** Média dos vetores $u, v$. Máximo de partículas (pior caso).
    * **POLIGONOS (Integral de Riemann):** $\sum (\text{Taxa kg/h} \times \Delta t)$. Resultado transformado em **Massa Total Emitida (kg)**.
    * **CARREGAMENTO:**
        * Soma de Toneladas (`Carregamento_Total`).
        * Moda do Produto (ou 'NO_OP').
        * Média de H2O (usando os valores imputados).
    * **CHUVA (Nova Agregação):** Soma de precipitação (se houver sensor) ou média de H2O (proxy).
3.  **Features Geométricas (Física):**
    * **Input:** Dicionário de coordenadas fixas (SOTA Precision).
        * **Praia (Target):** Lat **-20.7956661**, Lon **-40.5816175** (Ref: Ponto exato de coleta).
        * **RAMPs:** Lat/Lon extraídas do dicionário de configuração.
    * **Cálculo:**
        * Para cada RAMP $i$: Calcular $\theta_{praia}^i = \text{arctan2}(Lon_{praia}-Lon_i, Lat_{praia}-Lat_i)$.
        * `Fluxo_Efetivo` = $\sum (\text{Emissao}_i \times \cos(\text{Vento}_i - \theta_{praia}^i))$.
4.  **Output:** ABT Diária (`data/processed/abt_master.parquet`).

---

## 4. Estratégia de EDA (Robust Statistics & Physical Validation)

O EDA deve validar a física, a estatística e as **inconsistências de coleta**. Ordem de execução:

### Passo 1: Análise Univariada & Temporal
* **Line Chart:** Evolução do `PESO DO FILTRO` no tempo.
* **Histograma + KDE:** Verificar Log-Normalidade.

### Passo 2: Análise Bivariada Operacional (Teste A/B)
* **Box Plot Condicional:** Comparar distribuição de $y$ para `is_loading == 0` vs `is_loading == 1`.
* **Teste Estatístico:** Rodar **Mann-Whitney U Test** para validar se a diferença é significante (p-value < 0.05).

### Passo 3: Detecção de Anomalias (Chebyshev)
* **Objetivo:** Identificar eventos extremos sem assumir normalidade.
* **Ação:**
    * Calcular média ($\mu$) e desvio padrão ($\sigma$) de $y$.
    * Aplicar **Desigualdade de Chebyshev** ($k=3$ ou $4$).
    * Filtrar e listar dias onde $y > \mu + k\sigma$. Investigar se coincidem com ventania extrema.

### Passo 4: Análise Física (Windrose & Lags)
* **Windrose:** Plotar direção do vento ponderada pelo `PESO DO FILTRO`. O gráfico deve mostrar uma "pétala" na direção da empresa->praia.
* **Cross-Correlation (CCF):** Verificar correlação entre `Fluxo_Efetivo` e `Peso` com lags de 0 a 5 dias.

### Passo 5: Redução de Dimensionalidade (PCA de Sensores)
* **Objetivo:** Investigar redundância entre as alturas das RAMPs (9m, 16m, TSP).
* **Ação:** Padronizar -> PCA -> Scree Plot. Se PC1 explicar >90%, usar PC1 como proxy de poluição ambiental.

### Passo 6: Análise de "Washout" (Efeito da Chuva) - **NOVO**
* **Contexto:** A chuva pode "lavar" o filtro ou impedir a ressuspensão de poeira, gerando um peso artificialmente baixo mesmo com operação alta.
* **Ação:**
    * Criar feature `is_rainy` (Binária: chuva > limiar ou H20 > P75).
    * **Scatter Plot Colorido:** Eixo X: `Fluxo_Efetivo`, Eixo Y: `PESO DO FILTRO`, Cor: `is_rainy`.
    * **Hipótese:** Esperamos ver pontos de "Alta Emissão" mas "Baixo Peso" quando `is_rainy == True`.
    * **Teste de Robustez:** Aplicar **Mann-Whitney U** comparando o Peso em dias de "Alta Carga Seca" vs "Alta Carga Úmida". Se houver diferença significativa, a chuva é um regressor negativo obrigatório.

### Passo 7: Sensibilidade da Janela de Coleta (08:00 vs 09:00) - **NOVO**
* **Contexto:** A coleta ocorre entre 08:00 e 09:00, mas a unificação assume 09:00 fixo. Isso cria uma incerteza de 1 hora.
* **Ação:**
    * Calcular a variância (Std Dev) do Vento e Emissão especificamente na janela horária `08:00 - 09:00` de todos os dias.
    * **Plot:** Plotar `StdDev_Hora_Coleta` vs `Erro_Predicao` (Resíduo do Baseline).
    * **Hipótese:** Se os dias com alta volatilidade nessa hora tiverem maior erro no modelo, a imprecisão da coleta é a causa. Isso justifica o uso de modelos probabilísticos (que preveem um intervalo de confiança) em vez de determinísticos.

---

## 5. Estratégia de Modelagem (SOTA & Physics-Informed)

A modelagem deve priorizar robustez a outliers (distribuição Gamma) e causalidade temporal (Lags).

### A. Preparação de Features (Pré-Treino)
Antes de entrar no modelo, o script de treino deve gerar:
1.  **Lags Físicos (Crucial):**
    * `fluxo_efetivo_lag1` (O maior preditor segundo a EDA: r=0.48).
    * `fluxo_efetivo_lag2`.
    * `precipitacao_lag1` (A chuva de ontem afeta a poeira de hoje?).
2.  **Interaction Features:**
    * `interacao_vento_chuva` = `fluxo_efetivo` / (`precipitacao_mm` + 1).

### B. Configuração dos Modelos (Huber Loss Mandatório)
Devido à cauda longa (Outliers > 1.0g) e distribuição Gamma:

1.  **Baseline: XGBoost Regressor**
    * **Objective:** `reg:pseudohubererror` (ou `reg:gamma`).
    * **Eval Metric:** `mae` (para monitoramento) e `rmse` (para penalização).
    * **Handling Nulos:** O XGBoost lida nativamente, mas garantir que `is_loading` esteja presente.

2.  **SOTA: KAN (Kolmogorov-Arnold Network) / MLP**
    * **Loss Function:** `torch.nn.HuberLoss(delta=0.15)` (Delta baseado na mediana dos resíduos da EDA).
    * **Arquitetura:** Input -> [Lags + Física] -> KAN Layer -> Output (Relu para garantir não-negatividade).

### C. Estratégia de Validação
* **Split:** `TimeSeriesSplit` (5 Folds).
* **Gap:** Deixar 1 dia de gap entre treino e teste para evitar vazamento de lag.
* **Métrica de Sucesso Principal:** **MAE** (Erro Absoluto Médio) em gramas.
* **Check de Robustez:** O modelo deve errar menos nos dias de pico (Outliers) do que um modelo dummy (média).

### D. Artefatos de Saída
* Script `src/features/build_features_final.py` (Gera lags).
* Script `src/models/train_model.py` (Treina e loga métricas).
* Plot `reports/figures/08_model_predictions_vs_actual.png`.
* Plot `reports/figures/09_feature_importance.png` (Verificar se Lag1 está no topo).

---

## 6. Comandos Iniciais para o Claude Code
1.  Execute a FASE 1 (ETL) seguindo estritamente as regras de limpeza de strings e imputação.
2.  Gere a ABT em `data/processed/`.
3.  Execute a FASE 2 (EDA) gerando os gráficos, o relatório de anomalias de Chebyshev e os testes de hipótese de Washout.

## 7. Fase 3.D: Fine-Tuning & Otimização (Optuna)

**Objetivo:** Minimizar o MAE através da busca exaustiva de hiperparâmetros.

* **Script:** `src/models/tune_xgboost.py`.
* **Motor de Busca:** **Optuna** (100 trials).
* **Espaço de Busca:**
    * `max_depth`: [3, 10]
    * `learning_rate`: [0.01, 0.3]
    * `n_estimators`: [100, 1000]
    * `subsample` & `colsample_bytree`: [0.5, 1.0]
* **Regras de Ouro:**
    * Manter `objective='reg:pseudohubererror'`.
    * Validação cruzada obrigatória: `TimeSeriesSplit(n_splits=5, gap=1)`.
    * Salvar o melhor modelo em `models/xgb_optimized.json`.

## 8. Fase 4: Comunicação Técnica (Beamer Presentation)

**Objetivo:** Consolidar o workflow MLOps e os resultados finais.

* **Arquivo:** `reports/paper/presentation.tex`.
* **Conteúdo:** * Slides de infraestrutura, ETL e EDA.
    * Comparativo de modelos (XGBoost Baseline vs. Optimized vs. MLP vs. KAN).
    * Gráficos de importância de features e predição vs. real.