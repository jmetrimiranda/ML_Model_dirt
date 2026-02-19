# PROPOSAL V2: Arquitetura de Fluxo Efetivo Mássico Granular

## 1. O Problema da Abordagem Atual (V1)
O pipeline atual calcula o `fluxo_efetivo` fazendo a média do vento de *todas* as RAMPs da planta e multiplica pela emissão somada de *todos* os polígonos. 
**Erro Físico:** Isso cria um "falso transporte". Se a Torre C3 emitir muita massa, mas o vento local estiver soprando para o continente, o modelo V1 pode cruzar essa massa com o vento do Pátio (que pode estar soprando para a praia) e culpar erroneamente a operação. Sendo um problema de deposição de particulado (sujidade), o vetor inicial local é crucial.

## 2. A Solução SOTA (V2)
Calcular o transporte de massa localmente para cada fonte emissora antes da agregação, unindo a massa do polígono ($M_i$) estritamente com as medições de vento ($V_i$) das RAMPs ao seu redor.

**Fórmula do Fluxo Efetivo Mássico (kg*m/s) por Polígono:**
$$F_{massico\_i} = M_i \times \frac{1}{N_{ramps\_i}} \sum_{j=1}^{N_{ramps\_i}} \left( V_j \cdot \cos(\theta_{vento,j} - \theta_{praia,j}) \right)$$

## 3. Mapeamento Espacial (Polígono <-> RAMP)
O dicionário de configuração no ETL (`make_dataset.py`) deverá implementar a seguinte topologia:
- **Píer de Materiais:** RAMP 9, 10, 11, 12
- **Pátio de Estocagem:** RAMP 8, 14, 18, 19, 20, 21, 22
- **Usina 3:** RAMP 13, 14, 15, 16
- **Usina 4:** RAMP 4, 5, 6, 8
- **Torre C3:** RAMP 17, 18
*(Nota: RAMPs compartilhadas terão seus vetores eólicos utilizados no cálculo de ambos os polígonos correspondentes).*

## 4. Geração de Features para o Modelo
Em vez de um único `fluxo_emissao`, a ABT V2 terá 5 preditores diretos:
1. `fluxo_massico_pier`
2. `fluxo_massico_patio`
3. `fluxo_massico_usina3`
4. `fluxo_massico_usina4`
5. `fluxo_massico_torre_c3`

## 5. Explicabilidade e Causalidade Operacional
Ao treinar o XGBoost com essas 5 features desagregadas, aplicaremos:
- **SHAP Values:** Para identificar no nível diário qual equipamento foi responsável pela parcela majoritária da sujeira depositada na praia.
- **Permutation Feature Importance:** Para provar estatisticamente para a gestão qual polígono tem o maior impacto histórico na quebra de conformidade ambiental.