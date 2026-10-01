# Model card — Detecção de anomalias v1

| Campo | Valor |
|---|---|
| Modelo | `anomaly` |
| Versão | `v1` |
| Algoritmo | `IsolationForest` (scikit-learn 1.9.1), um modelo por tipo de ponto |
| Artefato | `artifacts/anomaly/v1/model.joblib` |
| Métricas completas | `artifacts/anomaly/v1/metrics.json` |
| Notebook | `notebooks/03-anomaly-detection.ipynb` |
| Endpoint | `POST /anomaly-score` |
| Treino | `uv run python -m ml.training.train --model anomaly` |

## Objetivo

Sinalizar sessões de recarga suspeitas antes do fechamento do rateio por kWh: energia impossível para o tempo conectado, leituras de medidor incoerentes e carros que ficam dias ocupando um ponto compartilhado. O síndico revisa as sessões sinalizadas.

## Como funciona

- Um Isolation Forest é treinado só com sessões normais de cada tipo de ponto (`PRIVATE` e `COMMERCIAL`). Cada sessão nova é avaliada pelo modelo do seu tipo.
- O score bruto do Isolation Forest é convertido em um **score normalizado de 0 a 1**, linear por partes e calculado por tipo de ponto:
  - 0,0 = percentil 1 dos scores de treino;
  - **0,5 = percentil 98 dos scores de treino (limiar)**;
  - 1,0 = score bruto 1,0.
- `isAnomaly = score >= 0,5`. Por construção, cerca de 2% das sessões normais são sinalizadas (equivale a `contamination = 0,02`).

Hiperparâmetros: 300 árvores, `max_samples = 1024`, semente 42.

## Entradas

| Campo da API | Variável |
|---|---|
| `energyKwh` | `energy_kwh` |
| `durationMinutes` | `duration_hours` |
| `idleMinutes` | `idle_hours` (limitado à duração) |
| `averagePowerKw` | `average_power_kw` |
| derivada | `charging_power_kw` = energia / (duração − ocioso) |
| `startHour` | `start_hour_sin`, `start_hour_cos` |
| `chargePointType` | escolhe o modelo (`PRIVATE` ou `COMMERCIAL`) |
| `dayOfWeek` | aceito pela API, não usado na v1 |

## Dados de treino

As mesmas fontes do fator de demanda (CC BY 4.0):

- Sørensen (2024), recarga residencial na Noruega, DOI 10.5281/zenodo.13896176;
- Andrenacci, Bosch e Kulla (2021), recarga pública da Turku Energia, DOI 10.5281/zenodo.5721233.

Foram usadas só as sessões válidas: energia ≥ 0,5 kWh, duração entre 5 minutos e 72 horas, potência dentro do limite físico e tempo ocioso conhecido. A divisão é aleatória: 31.939 sessões de treino e 7.956 de teste.

O tempo ocioso das sessões norueguesas é uma estimativa dos autores do conjunto de dados. Nas de Turku ele é estimado como `duração − energia / potência efetiva`, com 3,7 kW em AC e 37 kW em DC.

## Avaliação

Não existem rótulos reais de anomalia. A avaliação injeta 400 anomalias conhecidas (100 de cada tipo) nas 7.956 sessões normais de teste:

| Métrica (limiar 0,5) | Todos | `PRIVATE` | `COMMERCIAL` |
|---|---|---|---|
| Precisão | 0,66 | 0,63 | 0,77 |
| Cobertura (recall) | 0,80 | 0,77 | 0,94 |
| F1 | 0,72 | 0,70 | 0,85 |
| Sessões normais sinalizadas | 2,1% | 2,2% | 1,5% |
| ROC-AUC | 0,97 | 0,96 | 0,997 |

| Anomalia injetada | Cobertura |
|---|---|
| `meter_spike`: potência média 1,5 a 3 vezes acima do limite do ponto | 100% |
| `short_burst`: 20 a 60 kWh em 5 a 15 minutos | 100% |
| `extended_occupation`: conexão de 3 a 5 dias | 100% |
| `phantom_occupation`: 12 a 48 h conectado com menos de 1 kWh | 20% (71% em `COMMERCIAL`, 12% em `PRIVATE`) |

Sessões reais com mais de 72 horas, que ficaram fora do treino: **309 de 309 sinalizadas**.

## Limitações

- As anomalias de avaliação são sintéticas. A precisão real depende de quantas anomalias existem na operação; com anomalias mais raras, a fração de alertas falsos entre os sinalizados aumenta.
- Conectar o carro em casa e quase não carregar é comportamento comum nos dados residenciais, por isso `phantom_occupation` raramente é sinalizado em `PRIVATE`. Se o condomínio quiser coibir isso, o caminho é uma regra de tarifa por tempo ocioso.
- O tempo ocioso usado no treino é estimado.
- `dayOfWeek` não melhorou a detecção e ficou fora da v1.

## Uso recomendado e manutenção

- Usar o score para ordenar as sessões que o síndico deve revisar. Uma sessão sinalizada não deve ser cobrada nem bloqueada automaticamente.
- Registrar a decisão do síndico sobre cada alerta (procedente ou não). Esses rótulos permitem recalibrar o limiar e, no futuro, treinar um modelo supervisionado.
- Retreinar com as sessões do próprio condomínio quando houver histórico suficiente.
