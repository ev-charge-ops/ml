# EV ChargeOps — ML

Módulo de inteligência artificial do **EV ChargeOps**, plataforma de gestão de recarga de veículos elétricos em condomínios desenvolvida para o FIAP Enterprise Challenge 2026 com a GoodWe (Grupo 23, Sprint 02).

Este repositório contém o pipeline completo de dados, treino, avaliação e serviço de inferência dos dois modelos que compõem a IA estrutural da solução:

| Modelo | Algoritmo | Pergunta que responde | Endpoint |
|---|---|---|---|
| Fator de demanda | `GradientBoostingRegressor` | Quanto deve custar o kWh desta recarga, considerando a demanda das próximas horas? | `POST /demand-factor` |
| Detecção de anomalias | `IsolationForest` (um por tipo de ponto) | Esta sessão encerrada é suspeita e deve ser revisada pelo síndico antes do rateio? | `POST /anomaly-score` |

- **Serviço em produção:** https://ml.evchargeops.com.br (FastAPI na Vercel; documentação interativa em [`/docs`](https://ml.evchargeops.com.br/docs))
- **Stack:** Python 3.12, [uv](https://docs.astral.sh/uv/), scikit-learn 1.9.1, FastAPI, pandas e Jupyter (apenas no grupo `research`)

## Sumário

1. [Propósito dos modelos](#1-propósito-dos-modelos)
2. [Integração com a plataforma](#2-integração-com-a-plataforma)
3. [Conjuntos de dados](#3-conjuntos-de-dados)
4. [Métricas](#4-métricas)
5. [Notebooks](#5-notebooks)
6. [Contrato da API](#6-contrato-da-api)
7. [Como executar localmente](#7-como-executar-localmente)
8. [Estrutura do repositório](#8-estrutura-do-repositório)
9. [Limitações e próximos passos](#9-limitações-e-próximos-passos)

## 1. Propósito dos modelos

### 1.1 Fator de demanda (tarifa dinâmica por kWh)

No EV ChargeOps o custo de cada recarga é rateado por kWh medido. O fator de demanda é um multiplicador aplicado à tarifa base no momento em que o morador inicia a recarga:

```
preço por kWh = tarifa base × fator de demanda      (fator entre 0,8 e 1,5)
```

O objetivo é incentivar o uso dos carregadores compartilhados em horários de folga e desestimular a ocupação nos picos. O cálculo tem duas etapas separadas de propósito:

1. **Previsão (modelo).** Um `GradientBoostingRegressor` prevê o **índice de demanda das próximas 3 horas**: a média de carros conectados nas próximas 3 horas dividida pela capacidade do local. Vale 0 com a garagem vazia, 1 com ela na capacidade e pode passar de 1 quando há fila (limitado a 1,5). As entradas são as informações que o backend tem quando a recarga começa:

   | Campo da API | Variável do modelo | Tratamento |
   |---|---|---|
   | `hour` (0–23) | `hour` | — |
   | `dayOfWeek` (0 = segunda) | `day_of_week`, `is_weekend` | fim de semana = sábado e domingo |
   | `occupancyRatio` (0–1) | `occupancy_ratio` | carros conectados agora / capacidade |
   | `queueLength` (≥ 0) | `queue_length` | limitado a 10 |
   | `chargePointType` | `is_commercial` | `PRIVATE` = 0, `COMMERCIAL` = 1 |

   Hiperparâmetros: perda de Huber, 300 árvores, profundidade 5, taxa de aprendizado 0,05, subamostragem 0,8, semente 42 (escolhidos em validação separada do teste, notebook 02).

2. **Regra de preço (negócio).** Uma função linear por partes (`ml.models.demand_factor.demand_to_factor`) converte o índice previsto em fator, sempre limitado a [0,8; 1,5]:

   | Índice de demanda previsto | Fator | Interpretação |
   |---|---|---|
   | até 0,25 | 0,8 | fora de pico |
   | 0,50 | 1,0 | normal |
   | 0,90 ou mais | 1,5 | pico |

   Entre os pontos de controle o fator cresce linearmente. A regra é uma decisão de negócio: o síndico pode ajustá-la sem retreinar o modelo.

### 1.2 Detecção de anomalias nas sessões

Cada sessão vira cobrança para um morador. Uma sessão errada (medidor com defeito, energia impossível para o tempo conectado, carro abandonado dias no ponto compartilhado) gera cobrança injusta ou bloqueia o carregador. O detector sinaliza essas sessões para revisão do síndico **antes** do fechamento do rateio.

- **Por que não supervisionado:** não existe histórico de sessões rotuladas como erro ou fraude. O Isolation Forest aprende como é uma sessão normal e mede o quão fácil é isolar uma sessão nova das demais.
- **Um modelo por tipo de ponto** (`PRIVATE` e `COMMERCIAL`): sessões residenciais e públicas são muito diferentes (mediana de 11 h contra 2 h conectado). No notebook 03, um modelo único sinalizava 7,1% das sessões públicas normais contra 0,9% das residenciais; com um modelo por tipo, 1,5% e 2,2%.
- **Variáveis:** `energy_kwh`, `duration_hours`, `idle_hours` (limitado à duração), `average_power_kw`, `charging_power_kw` (energia / tempo efetivamente carregando) e a hora de início em forma circular (`start_hour_sin`, `start_hour_cos`). O `dayOfWeek` é aceito pela API, mas não é usado na v1.
- **Hiperparâmetros:** 300 árvores, `max_samples = 1024`, semente 42.

**Score e limiar.** O score bruto do Isolation Forest é convertido, por tipo de ponto, em um score normalizado de 0 a 1, linear por partes:

| Score normalizado | Âncora |
|---|---|
| 0,0 | percentil 1 dos scores brutos de treino |
| **0,5 (limiar)** | **percentil 98 dos scores brutos de treino** |
| 1,0 | score bruto 1,0 |

`isAnomaly = score >= 0,5`. Por construção, cerca de 2% das sessões normais são sinalizadas (equivale a `contamination = 0,02`). O score contínuo permite ao backend ordenar as sessões mais suspeitas primeiro.

**O que o detector encontra** (anomalias injetadas no teste, seção 4.2): picos de medidor (`meter_spike`), energia impossível em poucos minutos (`short_burst`) e carros abandonados por dias (`extended_occupation`) são sinalizados em 100% dos casos. Ocupação prolongada sem recarga (`phantom_occupation`) é a exceção, discutida nas [limitações](#9-limitações-e-próximos-passos).

## 2. Integração com a plataforma

```
 App (Expo) / Web (Vite)
          │
          ▼
   API NestJS ──── POST /demand-factor ──► ml.evchargeops.com.br (FastAPI)
   (módulo          POST /anomaly-score ─►   carrega artifacts/*/v1/model.joblib
   intelligence)                             na inicialização
```

A API NestJS ([ev-charge-ops/api](https://github.com/ev-charge-ops/api)) consome este serviço pelo módulo `intelligence`, configurado pela variável `ML_URL`, com timeout de 1,5 s por requisição:

| Momento | Chamada | Uso da resposta | Se o serviço de ML falhar |
|---|---|---|---|
| Consulta de preço dos pontos de recarga | `POST /demand-factor` com hora e dia da semana no fuso de São Paulo, ocupação, fila e tipo de ponto | O fator (arredondado a 2 casas) multiplica a tarifa base e é exposto com `source = MODEL` e a versão do modelo | **Fallback por regras** (`RuleDemandFactorProvider`): 1,5 se ocupação ≥ 80%, se houver fila ou entre 18h e 21h; 0,8 entre 22h e 6h com ocupação < 50%; 1,0 nos demais casos. A resposta indica `source = RULE` |
| Encerramento de uma sessão | `POST /anomaly-score` com energia, duração, tempo ocioso, potência média, hora de início, dia da semana e tipo de ponto | `score`, `isAnomaly` e `modelVersion` ficam gravados na sessão para revisão do síndico | A sessão é encerrada normalmente, sem score (o erro é registrado em log). O encerramento e a cobrança nunca dependem do serviço de ML |

Respostas fora do contrato (por exemplo, um fator fora de [0,5; 3] ou sem o campo `isAnomaly`) são tratadas pela API como falha e caem no mesmo caminho de contingência.

As variáveis são calculadas pelas mesmas funções de `ml.features` no treino e no serviço, o que evita diferenças entre treino e produção. Os artefatos carregados pela API são exatamente os avaliados nos notebooks 02 e 03.

