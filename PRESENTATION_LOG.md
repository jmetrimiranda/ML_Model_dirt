# PRESENTATION_LOG — Auditoria de Saude dos Dados (Raw → ABT)

Gerado em: 2026-02-18 | Auditoria para suporte ao Beamer final.

---

## 1. Tabela Y (Target)
- **Arquivo:** `data/raw/y_2025.csv`
- **Shape:** 365 linhas x 3 colunas (1 por dia, ano completo)
- **Frequencia:** Diaria
- **Colunas:** `Data`, `PESO DO FILTRO` (float), `CLASSE DO FILTRO` (float)
- **Nulos:**
  - `PESO DO FILTRO`: 47/365 (12.9%) → dias sem coleta
  - `CLASSE DO FILTRO`: 75/365 (20.5%)
- **Target:** min=0.0, median=0.01, mean=0.053, max=1.05g
- **Distribuicao:** Gamma (cauda longa). 75% dos dias < 0.05g, mas picos >1.0g existem.
- **Decisao ETL:** Remover dias com target NaN → 365 - 47 = 318 dias validos (316 na ABT final apos lags).

---

## 2. Tabela RAMPS (Meteorologia)
- **Arquivo:** `data/raw/ramps_2025.csv`
- **Shape:** 531.554 linhas x 13 colunas
- **Frequencia:** 15 minutos (sub-horaria)
- **Range:** 2025-01-01 00:07 a 2025-12-31 23:52
- **Estacoes:** Multiplas RAMPs (coluna `Origem_Arquivo`)
- **Nulos CRITICOS (% por coluna):**

| Coluna                         | Nulos (%) | Veredicto                          |
|--------------------------------|-----------|------------------------------------|
| Particulas 10.0m               | 94.0%     | **DESCARTADA** (quase total NaN)   |
| VelocidadeVento 6.0m           | 91.6%     | Fallback; nao e fonte primaria     |
| DirecaoVento 6.0m              | 91.6%     | Fallback; nao e fonte primaria     |
| DirecaoVento 10.0m             | 72.9%     | Fonte primaria (padrao ouro)       |
| VelocidadeVento 10.0m          | 72.7%     | Fonte primaria (padrao ouro)       |
| Particulas 2.0m                | 85.8%     | Descartada                         |
| Particulas 6.0m                | 85.8%     | Descartada                         |
| Particulas 3.0m                | 23.2%     | Utilizavel                         |
| Particulas 16.0m               | 23.2%     | **UTILIZADA** (melhor cobertura)   |
| Particulas 9.0m                | 17.2%     | **UTILIZADA** (melhor cobertura)   |

- **Outliers em Particulas:** max de 4416.6 ug/m3 (16.0m) e 3773.1 ug/m3 — possiveis tempestades de poeira ou falha de sensor.
- **Vento:** DirecaoVento 10.0m varia de 0 a 360 graus. VelocidadeVento 10.0m: max=23.3 m/s (plausivel).

---

## 3. Tabela POLIGONOS (Emissoes)
- **Arquivo:** `data/raw/poligonos_2025.csv`
- **Shape:** 138.144 linhas x 3 colunas
- **Frequencia:** 15 minutos
- **Range:** 2025-01-01 00:07 a 2025-12-31 23:52
- **Colunas:** `Data e Hora`, `Particulas [kg/h] 0.0 m` (float), `Origem_Arquivo`
- **Nulos:** Particulas: 3.287 (2.4%) — baixo e gerenciavel
- **5 Origens:** Torre C3, Usina 4, Usina3, Pier de materiais, Patio de estocagem
- **Problemas encontrados:**
  - Espaco inicial no nome da coluna: ` Particulas [kg/h] 0.0 m` (leading whitespace)
  - Inconsistencia de nomes: `Usina3` vs `Usina 4`
  - Nota: Nesta versao, a coluna ja foi parseada como float64 (sem virgulas como separador decimal nos dados atuais). O problema de casting "1,09" mencionado no TASK.md refere-se a versoes anteriores dos dados ou a um tratamento preventivo.
- **Cobertura esperada:** 5 fontes x 4/h x 24h x 365d = 175.200; real = 138.144 (79% — lacunas em algumas fontes).

---

## 4. Tabela CARREGAMENTO (Processo)
- **Arquivo:** `data/raw/carregamento_2025.csv`
- **Shape:** 8.762 linhas x 11 colunas
- **Frequencia:** Horaria (nao diaria como inicialmente assumido)
- **Range:** 2024-12-31 23:00 a 2026-01-01 00:00
- **Colunas:** `Dia / Hora`, `Carregamento (TMN)`, `Navio Atracado`, `Produto`, `H20 (%)`, `Finos -6,3mm (%)`, `Compressao 16,0(Kgf)`, `Compressao 12,5(Kgf)`, `Tempo Estoque (dias)`, `Descarga Pet Coke`, `Descarga Calcario`
- **Nulos SIGNIFICATIVOS:**

| Coluna                   | Nulos (%) | Causa                                 |
|--------------------------|-----------|---------------------------------------|
| Produto                  | 49.3%     | Horas sem navio                       |
| H20 (%)                  | 48.9%     | Horas sem navio                       |
| Finos -6,3mm (%)         | 66.4%     | Horas sem navio + coluna esparsa      |
| Compressao 16,0(Kgf)     | 66.4%     | Horas sem navio + coluna esparsa      |
| Compressao 12,5(Kgf)     | 66.4%     | Horas sem navio + coluna esparsa      |
| Tempo Estoque (dias)     | 48.9%     | Horas sem navio                       |

- **OUTLIERS CRITICOS em H2O:**
  - min=0.8%, max=**275.0%**, mean=5.52%
  - **3 valores > 100%** — fisicamente impossiveis (erro de digitacao: provavel deslocamento de virgula, ex: 2.75 → 275)
  - Valores > 25%: tambem apenas 3 linhas
  - **Tratamento:** Cap em 25% (TASK.md) ou NaN
- **Zeros em Carregamento:** 4.730 / 8.762 = **54.0%** das horas sem carga
  - Descarga Pet Coke: 92.8% zeros (evento raro)
  - Descarga Calcario: 96.5% zeros (evento raro)
- **Colunas com virgula decimal (object dtype):**
  - `Finos -6,3mm (%)` → precisa de str.replace(',', '.')
  - `Tempo Estoque (dias)` → precisa de str.replace(',', '.')

---

## 5. Tabela CHUVA (Precipitacao — Open-Meteo)
- **Arquivo:** `data/raw/chuva_2025.csv`
- **Shape:** 8.760 linhas x 3 colunas (365 x 24 = ano completo)
- **Frequencia:** Horaria
- **Range:** 2025-01-01 00:00 a 2025-12-31 23:00
- **Colunas:** `datetime`, `precipitacao_mm` (float), `chuva_mm` (float)
- **Nulos:** ZERO em todas as colunas
- **Redundancia:** `precipitacao_mm` e `chuva_mm` sao identicas (mesma estatistica)
- **Estatisticas:** mean=0.13 mm/h, max=16.5 mm/h, 75.8% zeros (maioria seco)
- **Tratamento:** Somar para diario; usar apenas 1 das 2 colunas redundantes.

---

## 6. ABT Final (Pos-ETL)
- **Arquivo:** `data/processed/abt_modeling.parquet`
- **Shape:** 316 linhas x 33 colunas
- **Conversao:** 365 dias → 318 (target valido) → 316 (perda por lags)
- **Nulos residuais:** Apenas `classe_filtro` com 29 NaN (9.2%). Demais: ZERO.
- **Target:** mean=0.0534g, median=0.01g, max=1.05g. 11 dias >0.3g (3.5%).
- **Features:** 5 de vento, 4 de particulas, 6 de massa/emissao, 2 de fluxo, 3 de chuva, 7 de carregamento, 3 de lags, 1 de interacao, 1 de classificacao.
