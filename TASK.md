# Contexto do Projeto: Predição de Sujidade em Praia (Mineração) - SOTA Architecture


## 0. Infraestrutura & Workflow (DevContainer + DVC + GitFlow)

**ATENÇÃO CLAUDE:** Este projeto roda em um ambiente estrito de MLOps. Antes de executar qualquer código ou sugerir alterações, você deve entender o estado atual do repositório.

### A. Estrutura de Arquivos (Project Tree)

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















## 1. Objetivo de Negócio
Prever a variável contínua `PESO DO FILTRO` (acumulado 24h) integrando dados meteorológicos de alta frequência (RAMPs), emissões calculadas de fluxo (Polígonos) e dados de processo portuário (Carregamento).
O modelo deve distinguir com precisão entre poeira gerada por operação ativa (Navio) vs. erosão eólica passiva (Pátio), lidando robustamente com a esparsidade dos dados operacionais.

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
3.  **Features Geométricas (Física):**
    * Calcular $\theta_{praia}$ (ângulo RAMP->Praia) usando `arctan2(delta_lon, delta_lat)`.
    * Calcular `Fluxo_Efetivo = Emissao_Total * cos(Vento_Medio - Theta_Praia)`.
4.  **Output:** ABT Diária (`data/processed/abt_master.parquet`).

---

## 4. Estratégia de EDA (Robust Statistics)

O EDA deve validar a física e a estatística dos dados unificados. Ordem de execução:

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

---

## 5. Estratégia de Modelagem (SOTA & Foundation Models)

A modelagem deve comparar abordagens clássicas, de ensemble e Deep Learning moderno.

### A. Validação
* **Método:** `TimeSeriesSplit` (5 Folds).
* **Restrição:** Proibido `Shuffle=True`. O treino deve ser sempre no passado do teste para evitar Data Leakage temporal.

### B. Modelos Tabulares (Baselines & SOTA)
1.  **Random Forest & Bagging:** Baseline robusto para entender a importância das features.
2.  **XGBoost / Gradient Boosting:** O estado da arte para dados tabulares com nulos. Lida nativamente com a flag `is_loading` e a não-linearidade do vento.
3.  **MLP (Multi-Layer Perceptron):** Rede neural densa simples como benchmark de Deep Learning.
4.  **KAN (Kolmogorov-Arnold Network):**
    * **Por que usar:** KANs modelam funções univariadas complexas (splines) nas arestas. Ideal para capturar a física exata da dispersão ($Vento^3$) sem precisar de feature engineering manual excessivo.

### C. Foundation Models de Série Temporal (Experimental)
* **Modelo:** **TimesFM (Google)** ou **Chronos**.
* **Discussão:**
    * Métodos clássicos (ARIMA) sofrem com regressores exógenos complexos (Vento, Navio).
    * Foundation Models pré-treinados em bilhões de pontos de dados podem capturar padrões sazonais e tendências melhor que modelos treinados do zero, especialmente com dataset pequeno (< 1 ano).
    * **Ação:** Usar TimesFM em modo *zero-shot* ou *fine-tuning* leve, passando as features exógenas (Vento, Carregamento) como contexto.

### D. Métricas de Sucesso
* **RMSE (Root Mean Squared Error):** Penaliza grandes erros (anomalias).
* **MAE (Mean Absolute Error):** Erro médio real em gramas.
* **R² Ajustado:** Quanto da variância da sujeira é explicada pelo modelo.

---

## 6. Comandos Iniciais para o Claude Code
1.  Execute a FASE 1 (ETL) seguindo estritamente as regras de limpeza de strings e imputação.
2.  Gere a ABT em `data/processed/`.
3.  Execute a FASE 2 (EDA) gerando os gráficos e o relatório de anomalias de Chebyshev.