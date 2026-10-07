# Model card — Fator de demanda v1

| Campo | Valor |
|---|---|
| Modelo | `demand_factor` |
| Versão | `v1` |
| Algoritmo | `GradientBoostingRegressor` (scikit-learn 1.9.1) |
| Artefato | `artifacts/demand_factor/v1/model.joblib` |
| Métricas completas | `artifacts/demand_factor/v1/metrics.json` |
| Notebook | `notebooks/02-demand-factor.ipynb` |
| Endpoint | `POST /demand-factor` |
| Treino | `uv run python -m ml.training.train --model demand_factor` |

## Objetivo

Calcular o multiplicador da tarifa por kWh no momento em que um morador inicia a recarga. O preço final é `tarifa base × fator`, com o fator entre 0,8 e 1,5. O objetivo é incentivar o uso dos carregadores compartilhados em horários de folga e desestimular a ocupação nos picos.

## Como funciona

1. O modelo prevê o **índice de demanda das próximas 3 horas**: média de carros conectados dividida pela capacidade do local. Vale 0 com a garagem vazia, 1 com ela na capacidade e pode passar de 1 quando há fila (limitado a 1,5).
2. Uma regra linear por partes converte o índice em fator:

| Índice previsto | Fator |
|---|---|
| até 0,25 | 0,8 (fora de pico) |
| 0,50 | 1,0 (normal) |
| 0,90 ou mais | 1,5 (pico) |

O resultado é sempre limitado a [0,8; 1,5]. A regra é uma decisão de negócio e pode ser ajustada sem retreinar o modelo.

## Entradas

| Campo da API | Variável | Tratamento |
|---|---|---|
| `hour` (0–23) | `hour` | — |
| `dayOfWeek` (0 = segunda) | `day_of_week`, `is_weekend` | fim de semana = sábado e domingo |
| `occupancyRatio` (0–1) | `occupancy_ratio` | carros conectados agora / capacidade |
| `queueLength` (≥ 0) | `queue_length` | limitado a 10 |
| `chargePointType` | `is_commercial` | `PRIVATE` = 0, `COMMERCIAL` = 1 |

Hiperparâmetros: perda de Huber, 300 árvores, profundidade 5, taxa de aprendizado 0,05, subamostragem 0,8, semente 42. A configuração foi escolhida em uma validação separada do teste (notebook 02, seção 2).

## Dados de treino

- `PRIVATE`: Sørensen, Å. L. (2024). *Electric vehicle charging dataset with 35,000 charging sessions from 12 residential locations in Norway*. Zenodo. DOI 10.5281/zenodo.13896176. Licença CC BY 4.0.
- `COMMERCIAL`: Andrenacci, N.; Bosch, R.; Kulla, A. (2021). Dados de recarga pública da Turku Energia (Finlândia, 2019). Zenodo. DOI 10.5281/zenodo.5721233. Licença CC BY 4.0.

As sessões foram convertidas em um painel hora a hora por local (228.572 horas). A divisão é temporal: as últimas 20% das horas de cada local formam o teste (182.869 horas de treino e 45.703 de teste).

## Desempenho no teste

| Grupo | Método | MAE | RMSE | R² |
|---|---|---|---|---|
| Todos | **modelo** | **0,093** | **0,176** | **0,774** |
| Todos | persistência (demanda futura = ocupação atual) | 0,096 | 0,213 | 0,671 |
| Todos | média histórica por tipo, hora e dia | 0,247 | 0,326 | 0,227 |
| `PRIVATE` | **modelo** | **0,068** | **0,100** | **0,898** |
| `PRIVATE` | persistência | 0,077 | 0,125 | 0,842 |
| `COMMERCIAL` | **modelo** | 0,113 | **0,220** | **0,611** |
| `COMMERCIAL` | persistência | 0,112 | 0,264 | 0,440 |

Importância das variáveis (impureza): ocupação 0,93; hora 0,05; tipo de ponto 0,01; demais abaixo de 0,01.

Verificações de comportamento (notebook 02 e `tests/test_models.py`): a demanda prevista não diminui quando a ocupação aumenta, e o fator fica sempre em [0,8; 1,5].

## Limitações

- Os dados são da Noruega e da Finlândia, não do Brasil.
- A capacidade de cada local foi estimada como o percentil 95 de carros conectados, e a fila é derivada dessa capacidade.
- No uso `COMMERCIAL` todas as estações de treino têm um único ponto, então a ocupação observada é 0 ou 1. Valores intermediários são interpolados pelo modelo.
- No uso `COMMERCIAL` o modelo empata com a persistência no erro absoluto médio. O ganho está nos momentos de mudança de demanda (RMSE e R²).
- A monotonicidade em relação à ocupação é verificada, mas não imposta no treino.

## Uso recomendado e manutenção

- Usar como sinal de preço para sessões iniciadas, nunca para recusar uma recarga.
- Exibir ao morador o fator aplicado antes de iniciar a sessão.
- Retreinar com o histórico do próprio condomínio após alguns meses de operação e comparar com esta versão usando as mesmas métricas.
