# TECH_MASTER_RECAP — Relatorio Tecnico Completo

Projeto: Predicao de Sujidade em Praia (Mineracao)
Autor: Jorge Metri | Gerado: 2026-02-18

---

## 1. MATEMATICA DE INTEGRACAO

### 1.1 Janela de Integracao (24 horas)

O filtro de coleta acumula particulas continuamente. A coleta ocorre diariamente
entre 08:00 e 09:00 (assumimos 09:00 como referencia fixa). Para cada dia $D$
com target valido, a janela de integracao e:

$$
W_D = [D_{-1} \text{ 09:00},\; D \text{ 09:00}]
$$

Todos os registros sub-horarios (RAMPs a cada 15 min, Poligonos a cada 15 min)
e horarios (Carregamento, Chuva) dentro dessa janela sao agregados em uma
unica linha diaria da ABT.

### 1.2 Integral de Riemann para Massa de Emissao (kg)

Os Poligonos reportam taxa de emissao em kg/h a cada 15 minutos. Para obter
a massa total emitida em 24h, aplicamos a Integral de Riemann discreta:

$$
M_{\text{total}} = \sum_{t \in W_D} \text{Taxa}(t) \times \Delta t
$$

Onde:
- $\text{Taxa}(t)$ = Particulas [kg/h] no timestamp $t$
- $\Delta t$ = intervalo entre medicoes (0.25h para dados de 15 min)
- $M_{\text{total}}$ = massa total emitida em kg no periodo de 24h

Isso e calculado separadamente para cada uma das 5 origens de emissao
(Torre C3, Usina 3, Usina 4, Pier de materiais, Patio de estocagem),
gerando as colunas `massa_torre_c3`, `massa_usina3`, `massa_usina4`,
`massa_pier`, `massa_patio` e a soma `emissao_total_kg`.

### 1.3 Media Vetorial do Vento (componentes u, v)

A direcao do vento em graus NAO pode ser mediada aritmeticamente (ex: media
de 350 e 10 graus daria 180, quando o correto e 0/360). A solucao e decompor
em componentes vetoriais:

$$
u = -V \cdot \sin\left(\frac{\pi \cdot \theta}{180}\right)
\qquad
v = -V \cdot \cos\left(\frac{\pi \cdot \theta}{180}\right)
$$

Onde:
- $V$ = VelocidadeVento [m/s] 10.0m (sensor padrao ouro)
- $\theta$ = DirecaoVento [graus] 10.0m
- $u$ = componente zonal (leste-oeste)
- $v$ = componente meridional (norte-sul)

A agregacao diaria calcula $\bar{u}$ e $\bar{v}$ (medias aritmeticas dos
componentes, que e valido por serem grandezas escalares). A direcao media
reconstituida e:

$$
\bar{\theta} = \text{arctan2}(-\bar{u}, -\bar{v}) \times \frac{180}{\pi}
$$

Tambem se calcula:
- `vento_vel_mean` = media de $V$ na janela
- `vento_vel_max` = maximo de $V$ na janela (rajada)

### 1.4 Agregacao do Carregamento

Para a tabela de Carregamento (frequencia horaria), a agregacao na janela de 24h
produz:
- `carreg_total_tmn` = $\Sigma$ Carregamento (TMN) — tonelagem total embarcada
- `produto_moda` = Moda do Produto (ou 'NO_OP' se nao houve navio)
- `h2o_media` = Media de H2O (%) usando valores imputados
- `tempo_estoque_media` = Media de Tempo Estoque (dias)
- `is_loading` = 1 se $\Sigma$ Carregamento > 0, else 0

### 1.5 Agregacao da Precipitacao

Para a tabela de Chuva (frequencia horaria, Open-Meteo):
- `precipitacao_mm` = $\Sigma$ precipitacao na janela de 24h
- `is_rainy` = 1 se precipitacao > limiar, else 0

---

## 2. FISICA DAS FEATURES

### 2.1 Fluxo Efetivo (Geometria Direcional)

O Fluxo Efetivo pondera a emissao de cada fonte pela componente do vento
na direcao da praia (ponto de coleta do filtro). A ideia fisica: so contribui
para sujidade a fracao da emissao que o vento efetivamente transporta
em direcao ao filtro.

**Coordenadas fixas (GPS de alta precisao):**
- Praia (filtro): Lat = -20.7956661, Lon = -40.5816175
- RAMPs: coordenadas do dicionario de configuracao

**Angulo praia-fonte:**

$$
\theta_{\text{praia}}^i = \text{arctan2}(\text{Lon}_{\text{praia}} - \text{Lon}_i,\; \text{Lat}_{\text{praia}} - \text{Lat}_i)
$$

**Fluxo Efetivo:**

$$
F_{\text{ef}} = \sum_i E_i \cdot \cos(\theta_{\text{vento},i} - \theta_{\text{praia}}^i)
$$

Onde:
- $E_i$ = emissao total (kg) da fonte $i$ na janela de 24h
- $\theta_{\text{vento},i}$ = direcao media do vento na RAMP $i$ (em radianos)
- $\theta_{\text{praia}}^i$ = angulo geometrico da fonte $i$ ate a praia
- $\cos(\cdot)$ = projecao: positivo quando o vento sopra DA fonte PARA a praia

Quando $\cos(\theta_v - \theta_p) > 0$: vento transporta poeira para a praia.
Quando $\cos(\theta_v - \theta_p) < 0$: vento afasta poeira da praia.

### 2.2 Interacao Vento/Chuva

A chuva suprime a ressuspensao de particulas. A feature de interacao normaliza
o fluxo efetivo pelo efeito supressor da precipitacao:

$$
I = \frac{F_{\text{ef}}}{P_{\text{mm}} + 1}
$$

Onde:
- $F_{\text{ef}}$ = Fluxo Efetivo (definido acima)
- $P_{\text{mm}}$ = precipitacao acumulada em mm na janela de 24h
- $+1$ no denominador evita divisao por zero e garante que dias secos
  ($P=0$) mantenham $I = F_{\text{ef}}$

Interpretacao: Chuva alta → denominador grande → interacao reduzida →
modelo "desconta" o efeito do vento.

### 2.3 Lags Temporais (Causalidade Fisica)

A deposicao de particulas no filtro e um processo cumulativo. A poeira emitida
ontem pode continuar sedimentando hoje. Lags capturam essa "memoria":

- `fluxo_efetivo_lag1`: Fluxo Efetivo do dia anterior ($r = 0.48$ com o target)
- `fluxo_efetivo_lag2`: Fluxo Efetivo de 2 dias atras
- `precipitacao_lag1`: Precipitacao do dia anterior (washout residual)

O lag1 do fluxo efetivo e consistentemente o preditor mais importante em
todos os modelos treinados (confirmado por Feature Importance e pela
Cross-Correlation, Fig. 4).

---

## 3. SAUDE DOS DADOS — DIAGNOSTICO COMPLETO

### 3.1 RAMPs (Meteorologia)

- **Arquivo:** `data/raw/ramps_2025.csv`
- **Volume:** 531.554 registros x 13 colunas
- **Frequencia:** 15 minutos
- **Periodo:** 2025-01-01 00:07 a 2025-12-31 23:52
- **Multiplas estacoes:** Identificadas pela coluna `Origem_Arquivo`

**Diagnostico de nulos por sensor:**

| Sensor                      | Nulos (%) | Decisao ETL              |
|-----------------------------|-----------|--------------------------|
| Particulas [ug/m3] 10.0m   | 94.0%     | DESCARTADA               |
| VelocidadeVento [m/s] 6.0m | 91.6%     | Fallback (secundario)    |
| DirecaoVento [°] 6.0m      | 91.6%     | Fallback (secundario)    |
| Particulas [ug/m3] 2.0m    | 85.8%     | DESCARTADA               |
| Particulas [ug/m3] 6.0m    | 85.8%     | DESCARTADA               |
| DirecaoVento [°] 10.0m     | 72.9%     | PADRAO OURO (primario)   |
| VelocidadeVento [m/s] 10.0m| 72.7%     | PADRAO OURO (primario)   |
| Particulas [ug/m3] 3.0m    | 23.2%     | Utilizavel               |
| Particulas [ug/m3] 16.0m   | 23.2%     | UTILIZADA                |
| Particulas [ug/m3] 9.0m    | 17.2%     | UTILIZADA (melhor cobert.)|

**Outliers:** Particulas 10.0m max = 4.416 ug/m3 (possivelmente tempestade
de poeira ou falha de sensor; valor tipico ambiente <150 ug/m3).
VelocidadeVento 10.0m: max = 23.3 m/s (plausivel para regiao costeira).

**Tratamento SOTA:**
1. Vento: Usar sensor 10.0m como primario; se NaN, imputar com 6.0m; senao, manter NaN.
2. Particulas: Usar 9.0m e 16.0m (maior cobertura). Descartar 10.0m (94% NaN).
3. Engenharia Vetorial: Converter graus → radianos → componentes u, v imediatamente.
   Proibida media aritmetica de graus.

### 3.2 Poligonos (Emissoes)

- **Arquivo:** `data/raw/poligonos_2025.csv`
- **Volume:** 138.144 registros x 3 colunas
- **Frequencia:** 15 minutos
- **Periodo:** 2025-01-01 00:07 a 2025-12-31 23:52

**5 origens de emissao:**
1. Torre C3
2. Usina3 (nota: sem espaco)
3. Usina 4 (nota: com espaco — inconsistencia)
4. Pier de materiais
5. Patio de estocagem

**Problemas encontrados:**
- Leading whitespace no nome da coluna: ` Particulas [kg/h] 0.0 m`
- Inconsistencia de nomes: `Usina3` vs `Usina 4`
- Nulos em emissao: 3.287 (2.4%) — baixo, gerenciavel
- Cobertura real: 138k de 175k esperados (79%) — lacunas em algumas fontes

**Tratamento SOTA:**
1. Strip whitespace dos nomes de coluna.
2. Normalizar nomes das origens.
3. Casting preventivo: `str.replace(',', '.')` → float (para versoes de dados
   com virgula decimal, como "1,09").
4. Pivot: Long → Wide format (1 coluna por origem).
5. Imputacao: Poligono ausente num timestamp = 0.0 (sem emissao calculada).
   NaN somente se TODOS os poligonos estiverem vazios (falha de sistema).

### 3.3 Carregamento (Processo Portuario)

- **Arquivo:** `data/raw/carregamento_2025.csv`
- **Volume:** 8.762 registros x 11 colunas
- **Frequencia:** Horaria
- **Periodo:** 2024-12-31 23:00 a 2026-01-01 00:00

**Venenos criticos identificados:**

1. **H2O (%) — Outliers absurdos:**
   - min = 0.8%, max = 275.0%, mean = 5.52%
   - 3 valores > 100% (fisicamente impossivel para umidade de minerio)
   - Causa provavel: erro de digitacao (deslocamento de virgula: 2.75 → 275)
   - Tratamento: Valores > 25% → NaN (clamping fisico)

2. **Colunas com virgula decimal (dtype object):**
   - `Finos -6,3mm (%)`: virgula como separador decimal
   - `Tempo Estoque (dias)`: virgula como separador decimal
   - Tratamento: `str.replace(',', '.')` → `pd.to_numeric()`

3. **Esparsidade operacional (zeros estruturais):**
   - Carregamento (TMN): 54.0% das horas = 0 (sem navio)
   - Descarga Pet Coke: 92.8% zeros (evento raro)
   - Descarga Calcario: 96.5% zeros (evento raro)
   - Nulos em Produto: 49.3%, H2O: 48.9%, Finos: 66.4% — todos estruturais
     (horas sem navio atracado nao geram dados de qualidade)

**Estrategia de Imputacao SOTA ("Carregamento Zero"):**

O problema: quando Carregamento = 0, as colunas H2O, Produto, Tempo Estoque
ficam nulas. Preencher com zero criaria um vies de "minerio seco" (H2O=0
sugere minerio extremamente seco, quando na verdade nao ha minerio sendo
processado).

Solucao em 3 acoes:
- **Acao 1 (Flag):** Criar feature binaria `is_loading` (1 se Carregamento > 0).
- **Acao 2 (Produto):** Preencher nulos com categoria string `'NO_OP'`.
- **Acao 3 (H2O e Estoque):** Preencher com **Rolling Median (janela 7 dias)**
  ou Mediana Global do Mes. Isso representa o "estado basal" das pilhas
  de minerio no patio quando nao ha navio — a umidade do minerio estocado
  nao desaparece so porque nao ha navio.

### 3.4 Chuva (Precipitacao — Open-Meteo)

- **Arquivo:** `data/raw/chuva_2025.csv`
- **Volume:** 8.760 registros x 3 colunas (365 x 24 = ano completo)
- **Frequencia:** Horaria
- **Periodo:** 2025-01-01 00:00 a 2025-12-31 23:00
- **Nulos:** 0 em todas as colunas (fonte externa confiavel)
- **Redundancia:** Colunas `precipitacao_mm` e `chuva_mm` sao identicas.
  Uma foi removida no ETL.
- **Zero-inflated:** 75.8% das horas = 0.0 mm (maioria seca)
- **Max:** 16.5 mm/h (evento de chuva forte)

### 3.5 Target (Y)

- **Arquivo:** `data/raw/y_2025.csv`
- **Volume:** 365 registros x 3 colunas (1 por dia)
- **Frequencia:** Diaria
- **Colunas:** `Data`, `PESO DO FILTRO` (float), `CLASSE DO FILTRO` (float)
- **Nulos no target:** 47/365 (12.9%) — dias sem coleta, removidos
- **Nulos em classe:** 75/365 (20.5%) — usada apenas para validacao, nao como feature
- **Distribuicao do target:**
  - min = 0.0g, Q25 = 0.0g, mediana = 0.010g, media = 0.053g
  - Q75 = 0.050g, max = 1.050g, std = 0.111g
  - 11 dias > 0.3g (3.5% — eventos extremos)
  - Distribuicao Gamma com cauda longa

### 3.6 ABT Final (Pos-ETL)

- **Arquivo:** `data/processed/abt_modeling.parquet`
- **Shape:** 316 linhas x 33 colunas
- **Conversao:** 365 dias → 318 (target valido) → 316 (perda por criacao de lags)
- **Nulos residuais:** Apenas `classe_filtro` com 29 NaN (9.2%). Demais: ZERO.
- **Compressao total:** 687k registros brutos → 316 linhas diarias

---

## 4. RESULTADOS DO BENCHMARK

### 4.1 Tabela Comparativa Completa

| Modelo                      | MAE (g) | RMSE (g) | Ganho vs Dummy | Parametros  |
|-----------------------------|---------|----------|----------------|-------------|
| Dummy (Media)               | 0.0634  | ---      | ---            | ---         |
| Random Forest               | 0.0546  | 0.0884   | +13.8%         | 500 arvores |
| LightGBM (Huber)            | 0.0522  | 0.0903   | +17.6%         | ---         |
| **XGBoost Baseline**        | **0.0499** | **0.0887** | **+21.3%** | **500 arvores** |
| XGBoost Optuna (100 trials) | 0.0501  | 0.0891   | +21.0%         | 900 arvores |
| MLP (Huber, delta=0.15)     | 0.0643  | 0.1097   | -1.4%          | 1.921       |
| KAN [39→16→1]               | 0.0918  | 0.1561   | -44.8%         | 9.578       |

**Metrica principal:** MAE (Erro Absoluto Medio) em gramas.
**Validacao:** TimeSeriesSplit com 5 folds e gap=1 dia (previne vazamento de lag).
**Loss function:** `reg:pseudohubererror` para XGBoost (robusta a outliers da cauda Gamma).

### 4.2 Por que o XGBoost Baseline Venceu

1. **Pseudo-Huber Loss:** Robusta a outliers (cauda longa >1.0g) sem descarta-los.
   A distribuicao Gamma do target faz com que MSE penalize excessivamente os picos,
   enquanto Huber Loss limita a influencia dos residuos extremos.

2. **Tratamento nativo de NaN:** XGBoost aprende a direcao otima dos splits quando
   encontra valores ausentes — nao precisa de imputacao explicita. Os 29 NaN
   residuais na ABT sao tratados automaticamente.

3. **Regularizacao intrinseca:** Arvores de decisao com profundidade limitada (max_depth)
   e early stopping (50 rounds) evitam overfitting natural em small data (316 amostras).

4. **Splits discretos:** A natureza discreta dos splits de arvore lida bem com
   a distribuicao zero-inflated do target (75% < 0.05g).

### 4.3 Por que o XGBoost Optuna NAO Superou o Baseline

O Optuna (100 trials, TPE Sampler) encontrou parametros que resultaram em
MAE = 0.0501g vs. baseline de 0.0499g. A diferenca e de +0.0002g — marginal
e dentro do ruido estatistico.

**Interpretacao:** Com 316 amostras e 33 features, a engenharia de features
(lags, fluxo efetivo, interacao vento/chuva) domina a performance. O tuning
de hiperparametros tem rendimentos decrescentes quando o feature space ja
captura a fisica do problema.

### 4.4 Por que a KAN Divergiu

A KAN (Kolmogorov-Arnold Network) com arquitetura [39→16→1] falhou com
MAE = 0.0918g (-44.8% PIOR que o dummy). Diagnostico:

1. **Overparametrizacao critica:**
   - 9.578 parametros para 316 amostras = ratio 30:1
   - Recomendado para redes neurais: >100:1 (idealmente >1000:1)
   - Resultado: overfitting nos dados de treino + underfitting nos picos de teste

2. **Distribuicao Gamma esparsa:**
   - 75% dos targets < 0.05g, mas picos atingem 1.05g
   - Gradientes instaveis: a rede aprende a "media" e ignora os extremos
   - Huber Loss (delta=0.15) atenua mas nao resolve com n tao pequeno

3. **NaN handling:**
   - Redes neurais exigem imputacao explicita de todos os NaN
   - Qualquer estrategia de imputacao introduz vies
   - XGBoost trata NaN nativamente com splits direcionais (sem vies)

4. **Suavizacao excessiva:**
   - KAN aprende funcoes suaves (base functions de B-splines)
   - A relacao vento → sujidade tem descontinuidades (ex: presenca/ausencia
     de navio e um switch binario, nao uma funcao suave)
   - XGBoost captura descontinuidades naturalmente via splits

5. **MLP tambem falhou (MAE = 0.0643g):**
   - Mesma classe de problemas: overparametrizacao + distribuicao Gamma
   - Com 1.921 params e 316 amostras, ratio ~6:1 (melhor que KAN, mas insuficiente)

---

## 5. MAPEAMENTO DE FIGURAS — O QUE CADA UMA PROVA

### Fig 01: `01_evolucao_temporal_peso.png`
**O que mostra:** Serie temporal do PESO DO FILTRO ao longo de 2025.
**O que prova:** A distribuicao Gamma do target — baseline proximo a zero com
picos esporadicos >0.6g. Evidencia sazonalidade e eventos extremos isolados.
Justifica o uso de Huber Loss (robusta a outliers).

### Fig 02: `02_bivariada_operacional.png`
**O que mostra:** Box plot condicional: PESO DO FILTRO para `is_loading=0` vs `is_loading=1`.
**O que prova:** Dias com navio atracado (carregamento ativo) apresentam distribuicao
de peso significativamente diferente de dias sem navio. Mann-Whitney U test
confirma (p < 0.05). A variavel `is_loading` e um discriminador legitimo
de regime operacional.

### Fig 03a: `03_windrose_poluicao.png`
**O que mostra:** Windrose ponderada pelo PESO DO FILTRO.
**O que prova:** A "petala" predominante aponta na direcao empresa → praia.
Isso confirma a CAUSALIDADE DIRECIONAL: o vento transporta particulas
da area de operacao para o ponto de coleta do filtro. Valida a feature
`fluxo_efetivo` e a geometria GPS usada.

### Fig 03b: `03_chebyshev_outliers.png`
**O que mostra:** Deteccao de anomalias via Desigualdade de Chebyshev (k=3).
**O que prova:** Os eventos extremos (y > mu + 3*sigma) coincidem com dias
de ventania intensa. Sem assumir normalidade (Chebyshev e distribution-free),
confirma que os outliers de peso sao FISICAMENTE EXPLICAVEIS — nao sao
erros de medicao. Justifica mante-los no dataset (nao remover).

### Fig 04: `04_cross_correlation.png`
**O que mostra:** Cross-correlation (CCF) entre Fluxo Efetivo e Peso do Filtro
com lags de 0 a 5 dias.
**O que prova:** Pico em lag 1, decaimento monotono ate lag 5. A "memoria"
do sistema e de ~24h — a poeira emitida ontem deposita hoje. Confirma
a importancia de `fluxo_efetivo_lag1` como feature e e consistente com
a fisica de deposicao gravitacional de particulas.

### Fig 05: `05_pca_scree_plot.png`
**O que mostra:** Scree plot do PCA aplicado aos sensores de particulas (9m, 16m).
**O que prova:** PC1 captura a maior parte da variancia entre alturas de sensor.
Os sensores de particulas a diferentes alturas sao altamente redundantes —
um componente principal e suficiente para representar a "poluicao ambiental"
global. Isso justifica nao incluir todos os sensores como features independentes.

### Fig 06: `06_efeito_chuva.png`
**O que mostra:** Scatter plot colorido por `is_rainy`. Eixo X: Fluxo Efetivo,
Eixo Y: PESO DO FILTRO.
**O que prova:** Dias chuvosos (pontos coloridos) com alta emissao apresentam
BAIXO peso. Confirma a hipotese de WASHOUT: a chuva "lava" as particulas
ou impede a ressuspensao, agindo como REGRESSOR NEGATIVO. Mann-Whitney U
test entre "Alta Carga Seca" vs "Alta Carga Umida" confirma (p < 0.05).
Justifica `precipitacao_mm` e `is_rainy` como features obrigatorias.

### Fig 07: `07_incerteza_coleta.png`
**O que mostra:** Std Dev do vento/emissao na janela 08-09h vs. Erro de Predicao
(residuo do baseline).
**O que prova:** Dias com alta volatilidade meteorologica na hora de coleta
apresentam RESIDUOS MAIORES. A imprecisao de +-1h na hora de coleta e uma
fonte de RUIDO IRREDUTIVEL. Justifica o uso futuro de modelos probabilisticos
(que preveem intervalos de confianca) em vez de deterministicos.

### Fig 08: `08_model_predictions_vs_actual.png`
**O que mostra:** Predicoes do XGBoost Baseline vs. valores reais ao longo do tempo.
**O que prova:** O modelo acompanha a tendencia geral do target. Dificuldade
nos picos extremos (>0.6g) — o modelo "subestima" os eventos extremos, o que
e esperado com Huber Loss (que limita a penalizacao de outliers).

### Fig 09: `09_feature_importance.png`
**O que mostra:** Feature Importance (Gain) do XGBoost Baseline — Top 15 features.
**O que prova:** `fluxo_efetivo_lag1` aparece consistentemente no topo. As features
de vento ($u$, $v$), emissao e precipitacao dominam. A FISICA DO TRANSPORTE
DE PARTICULAS e capturada pelo modelo. Features de processo (carregamento,
H2O) tambem aparecem, confirmando a relevancia operacional.

### Fig 10: `10_tree_benchmark.png`
**O que mostra:** Comparativo de MAE entre Random Forest, LightGBM e XGBoost.
**O que prova:** XGBoost com Pseudo-Huber Loss domina consistentemente em MAE
e RMSE. A robustez a outliers da Huber Loss e o diferencial — Random Forest
e LightGBM (com losses padrao) sao mais sensiveis a cauda longa.

### Fig 11: `11_kan_predictions_vs_actual.png`
**O que mostra:** Predicoes da KAN vs. valores reais.
**O que prova:** A KAN SUAVIZA os picos e DIVERGE nos extremos. Comportamento
tipico de overfitting + underfitting simultaneo: a rede memoriza o "regime
baixo" (75% dos dados < 0.05g) e falha nos eventos extremos. Visual
confirmacao do diagnostico de overparametrizacao (ratio 30:1).

### Fig 12: `12_model_comparison_kan_vs_xgb.png`
**O que mostra:** Comparacao lado a lado das predicoes KAN vs XGBoost.
**O que prova:** O gap entre os modelos CRESCE nos eventos extremos. XGBoost
mantem erro relativamente constante, enquanto KAN diverge exponencialmente.
Confirma que arvores de decisao sao mais adequadas que redes neurais
para este cenario de small data com distribuicao Gamma.

### Fig 13: `13_optuna_optimization.png`
**O que mostra:** Historico de otimizacao Optuna (100 trials, TPE Sampler).
**O que prova:** Convergencia rapida nas primeiras 20-30 trials, seguida de
plateau. O best MAE (0.0501g) e virtualmente igual ao baseline (0.0499g).
Confirma que ENGENHARIA DE FEATURES importa mais que TUNING DE HIPERPARAMETROS
neste cenario.

### Fig 14: `14_optimized_predictions.png`
**O que mostra:** Predicoes do XGBoost Optimizado (pos-Optuna) vs. real.
**O que prova:** Performance visual equivalente ao baseline. Picos >0.6g
permanecem como fronteira de melhoria — candidatos a modelagem probabilistica
(Quantile Regression) ou deteccao de regime.

### Fig 15: `15_optimized_feature_importance.png`
**O que mostra:** Feature Importance (Gain) do modelo otimizado — Top 15.
**O que prova:** CONSISTENCIA entre modelos: as mesmas features dominam
no baseline e no otimizado (vento_u, fluxo_emissao, precipitacao, lags).
A otimizacao de hiperparametros nao muda a ESTRUTURA do aprendizado —
apenas ajusta os pesos marginalmente. A fisica capturada e robusta.

---

## NOTA FINAL

A ABT de 316 dias x 33 features comprime 687k registros brutos em uma
representacao diaria fisicamente fundamentada. O XGBoost Baseline com
Pseudo-Huber Loss atinge MAE = 0.0499g (-21.3% vs. dummy), e as features
mais importantes refletem a fisica real do transporte de particulas:
vento direcional, emissao ponderada pela geometria, efeito supressor
da chuva, e a memoria de 24h do sistema (lag1).
