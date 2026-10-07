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

## 3. Conjuntos de dados

Dois conjuntos públicos, ambos com licença **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**:

| Tipo no EV ChargeOps | Fonte | Conteúdo |
|---|---|---|
| `PRIVATE` (garagem residencial compartilhada) | Sørensen, Å. L. (2024). *Electric vehicle charging dataset with 35,000 charging sessions from 12 residential locations in Norway*. Zenodo. DOI [10.5281/zenodo.13896176](https://doi.org/10.5281/zenodo.13896176) (artigo: Data in Brief, DOI [10.1016/j.dib.2024.110883](https://doi.org/10.1016/j.dib.2024.110883)) | ~35 mil sessões em 12 garagens residenciais na Noruega (2018–2021), com estimativa de tempo ocioso |
| `COMMERCIAL` (uso público/rotativo) | Andrenacci, N.; Bosch, R.; Kulla, A. (2021). *Supporting data to the paper "Modelling charge profiles of electric vehicles based on charges data"*. Zenodo. DOI [10.5281/zenodo.5721233](https://doi.org/10.5281/zenodo.5721233) | ~7,7 mil sessões em 18 estações públicas da Turku Energia (Finlândia, 2019), AC 22 kW e DC 50 kW |

### Download

```bash
uv run python -m ml.data.download          # baixa para data/raw/ e confere o MD5 publicado no Zenodo
uv run python -m ml.data.download --force  # baixa de novo mesmo se os arquivos já existirem
```

São baixados três arquivos (`norway_charging_reports.csv`, `norway_session_predictions.csv` e `turku_public_charging_2019.csv`) para `data/raw/`, que fica fora do Git. Uma amostra com 1.686 sessões de 5 locais (setembro e outubro de 2019) fica versionada em `data/sample/sessions.csv` e é usada pelos testes automatizados, que portanto não dependem do download.

### Preparação

- **Formato único** (`ml.data.loaders.load_sessions`): uma linha por sessão, com local, tipo de ponto, conexão, desconexão, energia, duração e tempo ocioso.
- **Sessões válidas** (`is_valid_session`): energia ≥ 0,5 kWh, duração entre 5 minutos e 72 horas, potência média dentro do limite físico do ponto (22 kW residencial, 50 kW público) e tempo ocioso conhecido.
- **Painel hora a hora** (`ml.data.hourly.build_hourly_panel`): carros conectados minuto a minuto por local, convertidos em ocupação no início de cada hora, fila estimada e índice de demanda. Semanas sem nenhuma sessão são removidas (falha de coleta).

### Ressalvas

- **Dados nórdicos.** Os dados são da Noruega e da Finlândia, não do Brasil. O padrão de horário (chegar em casa à noite e carregar de madrugada) tende a se repetir, mas clima, tarifas e frota são diferentes.
- **Capacidade e fila estimadas.** Nenhuma fonte informa quantos carregadores existem nem registra fila. A capacidade de cada local é o percentil 95 da média horária de carros conectados (arredondado para cima), e a fila é o excedente de carros conectados sobre essa capacidade.
- **Tempo ocioso estimado.** Nas sessões norueguesas ele é uma estimativa dos autores do conjunto de dados. Nas de Turku é calculado como `duração − energia / potência efetiva`, com 3,7 kW em AC e 37 kW em DC (percentil 90 da potência observada). Carros mais rápidos que isso ficam com ocioso zero, então a estimativa é conservadora.
- **Estações públicas de um ponto.** No uso `COMMERCIAL` todas as estações têm capacidade estimada igual a 1, então a ocupação observada é sempre 0 ou 1.

## 4. Métricas

Todos os números abaixo vêm de `artifacts/*/v1/metrics.json`, gerados por `ml.training.train` sobre os artefatos versionados que a API carrega. Detalhes em cada model card: [fator de demanda](artifacts/demand_factor/v1/model-card.md) e [anomalias](artifacts/anomaly/v1/model-card.md).

### 4.1 Fator de demanda

Painel de 228.572 horas. Divisão **temporal**: as últimas 20% das horas de cada local formam o teste (182.869 horas de treino e 45.703 de teste). Uma divisão aleatória vazaria informação entre horas vizinhas.

O modelo é comparado com dois baselines:

- **Persistência:** a demanda das próximas horas é igual à ocupação atual.
- **Média histórica:** média do alvo no treino para o mesmo tipo de ponto, hora e dia da semana.

| Grupo | Método | MAE | RMSE | R² |
|---|---|---|---|---|
| Todos | **modelo** | **0,093** | **0,176** | **0,774** |
| Todos | persistência | 0,096 | 0,213 | 0,671 |
| Todos | média histórica | 0,247 | 0,326 | 0,227 |
| `PRIVATE` | **modelo** | **0,068** | **0,100** | **0,898** |
| `PRIVATE` | persistência | 0,077 | 0,125 | 0,842 |
| `PRIVATE` | média histórica | 0,248 | 0,306 | 0,053 |
| `COMMERCIAL` | **modelo** | 0,113 | **0,220** | **0,611** |
| `COMMERCIAL` | persistência | **0,112** | 0,264 | 0,440 |
| `COMMERCIAL` | média histórica | 0,247 | 0,342 | 0,064 |

Leitura:

- O modelo supera a média histórica por larga margem em todos os grupos: saber hora e dia não basta, é preciso olhar o estado atual.
- Contra a persistência, reduz o RMSE em 17% no conjunto (0,176 contra 0,213) e melhora o R² nos dois tipos de ponto. O ganho está nos momentos de transição (chegada dos moradores no fim da tarde, saída pela manhã), que são os que importam para o preço.
- No uso `COMMERCIAL` o modelo empata com a persistência no MAE (0,113 contra 0,112).
- Importância das variáveis (impureza): ocupação 0,93; hora 0,05; tipo de ponto 0,01; dia da semana, fim de semana e fila abaixo de 0,01.
- Distribuição do fator no teste: percentis 10, 25 e 50 em 0,80; percentil 75 em 1,18; percentil 90 em 1,45.
- Verificações de comportamento (notebook 02 e `tests/test_models.py`): a demanda prevista não diminui quando a ocupação aumenta, e o fator fica sempre em [0,8; 1,5].

### 4.2 Detecção de anomalias

Divisão aleatória 80/20 sobre as sessões válidas: 31.939 sessões de treino e 7.956 de teste. Como não há rótulos reais, a avaliação injeta **400 anomalias conhecidas** (100 de cada tipo, semente fixa) nas sessões normais de teste:

| Tipo injetado | Como é gerado | Cobertura |
|---|---|---|
| `meter_spike` | potência média 1,5 a 3 vezes acima do limite físico do ponto | 100% |
| `short_burst` | 20 a 60 kWh entregues em 5 a 15 minutos | 100% |
| `extended_occupation` | conexão estendida para 3 a 5 dias, sem energia adicional | 100% |
| `phantom_occupation` | 12 a 48 h conectado com menos de 1 kWh | 20% |

Métricas no limiar 0,5:

| Métrica | Todos | `PRIVATE` | `COMMERCIAL` |
|---|---|---|---|
| Precisão | 0,66 | 0,63 | 0,77 |
| Cobertura (recall) | 0,80 | 0,77 | 0,94 |
| F1 | 0,72 | 0,70 | 0,85 |
| Sessões normais sinalizadas | 2,1% | 2,2% | 1,5% |
| ROC-AUC | 0,97 | 0,96 | 0,997 |
| Precisão média (AP) | 0,75 | 0,69 | 0,94 |

Comparação com alternativas (notebook 03):

- **Modelo único com o tipo de ponto como variável** (baseline): sinaliza 7,1% das sessões públicas normais e 0,9% das residenciais, contra 1,5% e 2,2% com um modelo por tipo, que também tem cobertura um pouco maior nos dois tipos.
- **Limiar:** o percentil 95 encontraria 85% das anomalias com precisão de 45%, e o percentil 99,5 teria 82% de precisão, mas deixaria passar metade das anomalias. O percentil 98 equilibra volume de revisão (cerca de 2 a cada 100 sessões) e cobertura (80%).

**Validação com sessões reais fora do treino:** as 309 sessões reais com mais de 72 horas, excluídas do treino, foram **todas sinalizadas** (309 de 309).

## 5. Notebooks

Os notebooks ficam em `notebooks/`, estão versionados já executados (com saídas e gráficos) e usam as mesmas funções do pacote `ml`.

| Notebook | Conteúdo |
|---|---|
| [`01-exploratory-analysis.ipynb`](notebooks/01-exploratory-analysis.ipynb) | Fontes de dados e regras de qualidade; distribuições de energia, duração e tempo ocioso (72% do tempo conectado é ocioso na sessão residencial mediana); mapa de calor dos horários de início por tipo de ponto; construção do painel hora a hora com ocupação e fila; correlação entre ocupação atual e demanda das próximas 3 horas, que define o desenho do modelo de demanda |
| [`02-demand-factor.ipynb`](notebooks/02-demand-factor.ipynb) | Montagem do alvo e divisão temporal; escolha de hiperparâmetros em validação separada (perda quadrática ou Huber, profundidade 3 ou 5); avaliação do artefato v1 contra persistência e média histórica; importância por impureza e por permutação; grade de cenários por hora e ocupação; regra índice → fator e distribuição do fator no teste |
| [`03-anomaly-detection.ipynb`](notebooks/03-anomaly-detection.ipynb) | Variáveis e justificativa do modelo não supervisionado; normalização do score e limiar; avaliação com anomalias injetadas por tipo; comparação com um modelo único; escolha do limiar (percentis 95, 98 e 99,5); sessões reais com mais de 72 horas e sessões normais com maior score |

Para abri-los localmente:

```bash
uv sync --all-groups
uv run python -m ml.data.download
uv run jupyter lab notebooks/
```

