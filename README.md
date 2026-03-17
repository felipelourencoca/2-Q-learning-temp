# Q-Learning para Maquina de Estados Finitos (FSM)

Implementacao de um agente **Q-Learning** aplicado a uma **Maquina de Estados Finitos (FSM)** carregada a partir de **arquivos JSON externos**. O agente aprende, por aprendizado por reforco, a navegar entre os estados da FSM.

Os testes de **State Coverage** sao executados automaticamente contra todos os arquivos JSON disponiveis.

---

## Indice

- [Visao Geral](#visao-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Conceitos Teoricos](#conceitos-teoricos)
- [Formato do JSON de Entrada](#formato-do-json-de-entrada)
- [Hiperparametros do Q-Learning](#hiperparametros-do-q-learning)
- [Como Executar](#como-executar)
- [Testes - State Coverage](#testes--state-coverage)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Referencias](#referencias)

---

## Visao Geral

Este projeto demonstra a aplicacao do algoritmo **Q-Learning** - um metodo de aprendizado por reforco *model-free* - para navegar em uma Maquina de Estados Finitos. A FSM e definida em um **arquivo JSON externo**, que e lido e convertido automaticamente para a estrutura interna do projeto.

### Objetivos do projeto

- Carregar FSMs a partir de **arquivos JSON externos**.
- Implementar um agente Q-Learning com politica epsilon-greedy.
- Treinar o agente para explorar e aprender caminhos na FSM.
- Validar os resultados com testes automatizados baseados no criterio de **State Coverage**.

---

## Estrutura do Projeto

```
Q-learning-fst/
|-- fsm.py                          # Definicao da classe FSM generica
|-- fsm_loader.py                   # Carregamento de FSM a partir de JSON
|-- q_learning.py                   # Implementacao do agente Q-Learning
|-- random_agent.py                 # Agente aleatorio (baseline)
|-- main.py                         # Script principal (treina e demonstra)
|-- analyze.py                      # Analise quantitativa e graficos
|-- test_state_coverage.py          # Suite de testes parametrizados (pytest)
|-- conftest.py                     # Opcoes customizadas do pytest
|-- requirements.txt                # Dependencias do projeto
|-- README.md                       # Este arquivo
|-- results/                        # Graficos gerados pelo analyze.py
|-- finite_states_machines/         # Arquivos JSON com FSMs
    |-- _01_LightSwitch_flattened.json
    |-- _02_DimmableLightSwitch_flattened.json
    |-- _03_MotionLightSwitch_flattened.json
    |-- _04_LightAndMotionSensingLightSwitch_flattened.json
    |-- _05_PresenceSimulationLightSwitch_flattened.json
```

| Arquivo | Descricao |
|---------|-----------|
| `fsm.py` | Classe `FSM` - encapsula estados, acoes, transicoes, recompensas, estados-objetivo e estado inicial. |
| `fsm_loader.py` | Funcao `load_fsm_from_json()` - le um JSON e converte para `FSM`. Inclui automaticamente estados-destino referenciados em transicoes e ignora transicoes com `input: null`. |
| `q_learning.py` | Classe `QLearningAgent` - treinamento, Q-table, politica epsilon-greedy, extracao de caminhos otimos e metricas de cobertura por episodio. |
| `random_agent.py` | Classe `RandomAgent` - agente baseline que escolhe acoes aleatorias (sem aprendizado), para comparacao com o Q-Learning. |
| `main.py` | Carrega a FSM do JSON, treina o agente e imprime os caminhos aprendidos. |
| `analyze.py` | Gera graficos comparativos (Q-Learning vs Random): curva de aprendizado, convergencia de cobertura, sensibilidade de hiperparametros e eficiencia de caminho. |
| `test_state_coverage.py` | Testes parametrizados que auto-descobrem todos os JSONs e verificam State Coverage, Transition Coverage e Transition Pair Coverage. |
| `conftest.py` | Registra opcoes customizadas de CLI para o pytest (hiperparametros). |

---

## Conceitos Teoricos

### Maquina de Estados Finitos (FSM)

Uma FSM e definida formalmente pela 5-tupla **M = (S, A, delta, R, G)**:

| Simbolo | Significado |
|---------|-------------|
| **S** | Conjunto finito de estados |
| **A** | Conjunto finito de acoes |
| **delta** | Funcao de transicao: delta(s, a) -> s' |
| **R** | Funcao de recompensa: R(s, a) -> r |
| **G** subconjunto de S | Conjunto de estados-objetivo (terminais) |

### Q-Learning

Algoritmo de aprendizado por reforco *off-policy* e *model-free*. Mantem uma tabela **Q(s, a)** com a regra de atualizacao:

```
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

### Politica epsilon-Greedy

- Com probabilidade **epsilon**: acao aleatoria (exploracao)
- Com probabilidade **1 - epsilon**: melhor acao conhecida (exploitacao)

O epsilon decai ao longo do treinamento (exploracao -> exploitacao gradual).

---

## Formato do JSON de Entrada

```json
{
  "initial": "nome_do_estado_inicial",
  "states": [
    {
      "state": "nome_do_estado",
      "transitions": [
        {
          "input": "nome_da_acao",
          "output": ["efeito1"],
          "target": "nome_do_estado_destino"
        }
      ]
    }
  ]
}
```

| Campo | Descricao |
|-------|-----------|
| `initial` | Estado inicial da FSM |
| `states[].state` | Nome do estado |
| `transitions[].input` | Acao que dispara a transicao (`null` e ignorado) |
| `transitions[].output` | Efeitos da transicao (informativo) |
| `transitions[].target` | Estado destino |

> **Nota:** O loader inclui automaticamente estados que sao referenciados como `target` mas nao estao listados explicitamente no array `states`.

---

## Hiperparametros do Q-Learning

Todos os hiperparametros sao configuraveis via linha de comando, tanto no `main.py` quanto no `test_state_coverage.py`.

| Parametro | Flag CLI | Padrao | Descricao |
|-----------|----------|:------:|-----------|
| alpha | `--alpha` | 0.1 | Taxa de aprendizado |
| gamma | `--gamma` | 0.9 | Fator de desconto |
| epsilon | `--epsilon` | 1.0 | Taxa de exploracao inicial |
| epsilon_decay | `--epsilon-decay` | 0.995 | Decaimento do epsilon por episodio |
| epsilon_min | `--epsilon-min` | 0.01 | Epsilon minimo |
| episodes | `--episodes` | 1000 | Episodios de treinamento (apenas main.py) |
| max_steps | `--max-steps` | 50 | Limite de passos por episodio |

---

## Como Executar

### Pre-requisitos

- Python 3.8+
- pip

### Instalacao

```bash
git clone https://github.com/seu-usuario/Q-learning-fst.git
cd Q-learning-fst
pip install -r requirements.txt
```

### Treinamento do Q-Learning (main.py)

Carrega uma FSM de um arquivo JSON e treina o agente:

```bash
# Minimo (apenas json obrigatorio)
python main.py --json finite_states_machines/_01_LightSwitch_flattened.json

# Com estado-objetivo
python main.py --json finite_states_machines/_02_DimmableLightSwitch_flattened.json --goal main_Off__brightness_10

# Com multiplos estados-objetivo
python main.py --json finite_states_machines/_02_DimmableLightSwitch_flattened.json --goal main_Off__brightness_5 main_Off__brightness_10

# Ajustando episodios e passos
python main.py --json finite_states_machines/_03_MotionLightSwitch_flattened.json --episodes 500 --max-steps 100

# Configurando hiperparametros do Q-Learning
python main.py --json finite_states_machines/_01_LightSwitch_flattened.json --alpha 0.2 --gamma 0.8

# Tudo junto
python main.py --json finite_states_machines/_02_DimmableLightSwitch_flattened.json --goal main_Off__brightness_10 --episodes 2000 --max-steps 200 --alpha 0.15 --gamma 0.95 --epsilon 0.8 --epsilon-decay 0.99 --epsilon-min 0.05

# FSM grande com poucos episodios
python main.py --json finite_states_machines/_05_PresenceSimulationLightSwitch_flattened.json --episodes 100 --max-steps 30
```

#### Argumentos do main.py

| Argumento | Obrigatorio | Tipo | Padrao | Descricao |
|-----------|:-----------:|------|:------:|-----------|
| `--json` | **Sim** | string | - | Caminho para o JSON da FSM |
| `--goal` | Nao | string(s) | `None` | Estado(s)-objetivo (separados por espaco) |
| `--episodes` | Nao | int | `1000` | Numero de episodios de treinamento |
| `--max-steps` | Nao | int | `50` | Maximo de passos por episodio |
| `--alpha` | Nao | float | `0.1` | Taxa de aprendizado |
| `--gamma` | Nao | float | `0.9` | Fator de desconto |
| `--epsilon` | Nao | float | `1.0` | Taxa de exploracao inicial |
| `--epsilon-decay` | Nao | float | `0.995` | Decaimento do epsilon |
| `--epsilon-min` | Nao | float | `0.01` | Epsilon minimo |

---

## Testes - Criterios de Cobertura

Os testes sao **parametrizados** e auto-descobrem todos os JSONs em `finite_states_machines/`. Cada teste e executado uma vez para cada FSM encontrada.

Tres criterios de cobertura sao implementados, em ordem crescente de rigor:

1. **State Coverage** - Todos os estados foram visitados?
2. **Transition Coverage** - Todas as transicoes foram exercitadas?
3. **Transition Pair Coverage** - Todos os pares de transicoes consecutivas foram cobertos?

### Classes de teste

| Classe | Usa Q-Learning? | Criterio | O que testa |
|--------|:-:|:-:|---|
| `TestStateCoverage` | Nao | State Coverage | Estrutura da FSM: estados existem, transicoes validas, alcancabilidade via BFS |
| `TestStateCoverageQLearning` | Sim | State Coverage | Agente explorou os estados e Q-table foi populada |
| `TestStateCoverageReport` | Sim | State Coverage | Gera relatorio de cobertura com os caminhos aprendidos |
| `TestTransitionCoverage` | Ambos | Transition Coverage | Todas as transicoes sao alcancaveis (BFS) e exercitadas pelo Q-Learning |
| `TestTransitionPairCoverage` | Sim | Transition Pair Coverage | Pares consecutivos de transicoes foram cobertos |

### Executar os testes

```bash
# Todos os criterios em todas as FSMs (automatico)
python -m pytest test_state_coverage.py -v
```

### Filtrar por FSM

```bash
# Apenas uma FSM especifica
python -m pytest test_state_coverage.py -v -k "_01"
python -m pytest test_state_coverage.py -v -k "_02"

# Varias FSMs
python -m pytest test_state_coverage.py -v -k "_01 or _03"

# Excluir uma FSM
python -m pytest test_state_coverage.py -v -k "not _05"
```

### Filtrar por criterio de cobertura

```bash
# Apenas State Coverage (sem Q-Learning)
python -m pytest test_state_coverage.py -v -k "TestStateCoverage and not QLearning and not Report"

# Apenas Transition Coverage
python -m pytest test_state_coverage.py -v -k "TransitionCoverage and not Pair"

# Apenas Transition Pair Coverage
python -m pytest test_state_coverage.py -v -k "TransitionPairCoverage"

# Apenas relatorios (todos os criterios)
python -m pytest test_state_coverage.py -v -s -k "Report or report"
```

### Combinar filtros (criterio + FSM)

```bash
# Transition Coverage apenas na maquina _03
python -m pytest test_state_coverage.py -v -k "TransitionCoverage and _03"

# Relatorio de cobertura de estados na maquina _01
python -m pytest test_state_coverage.py -v -s -k "Report and _01"

# Todos os relatorios da maquina _02
python -m pytest test_state_coverage.py -v -s -k "(_02) and (Report or report)"
```

### Configurar hiperparametros nos testes

```bash
# Ajustar passos
python -m pytest test_state_coverage.py -v -k "_01" --max-steps 30

# Ajustar taxa de aprendizado e desconto
python -m pytest test_state_coverage.py -v -k "_02" --alpha 0.2 --gamma 0.8

# Tudo junto
python -m pytest test_state_coverage.py -v -s -k "Report and _03" --max-steps 100 --alpha 0.15 --gamma 0.95 --epsilon 0.5 --epsilon-decay 0.99 --epsilon-min 0.05
```

### Flags uteis do pytest

| Flag | Descricao |
|------|-----------|
| `-v` | Mostra cada teste individualmente (verbose) |
| `-s` | Exibe os `print()` dos testes (relatorio de cobertura) |
| `-k "filtro"` | Filtra por nome do teste/arquivo |

### FSMs disponiveis

| Arquivo | Descricao | Estados |
|---------|-----------|:-------:|
| `_01_LightSwitch` | Interruptor simples (On/Off) | 2 |
| `_02_DimmableLightSwitch` | Interruptor com dimmer | 21 |
| `_03_MotionLightSwitch` | Luz com sensor de movimento | ~40 |
| `_04_LightAndMotionSensing...` | Luz + sensor de movimento/luminosidade | ~50 |
| `_05_PresenceSimulation...` | Simulacao de presenca | Centenas |

---

## Analise Quantitativa (analyze.py)

O script `analyze.py` gera graficos para a dissertacao, comparando **Q-Learning** com um **Random Agent** (baseline).

### Executar a analise

```bash
# Analisar uma FSM especifica
python analyze.py --json finite_states_machines/_01_LightSwitch_flattened.json

# Analisar todas as FSMs
python analyze.py --all-fsms

# Configurar episodios e passos
python analyze.py --json finite_states_machines/_02_DimmableLightSwitch_flattened.json --episodes 2000 --max-steps 100
```

### Graficos gerados (salvos em `results/`)

| Grafico | Descricao |
|---------|----------|
| `learning_curve_*.png` | Recompensa acumulada × episodios (Q-Learning vs Random) |
| `coverage_convergence_*.png` | State Coverage e Transition Coverage × episodios |
| `sensitivity_*.png` | Impacto de alpha, gamma e epsilon na cobertura |
| `path_efficiency_*.png` | Comprimento dos caminhos Q-Learning vs BFS otimo |

---

## Tecnologias Utilizadas

- **Python 3** - Linguagem principal
- **NumPy** - Computacao numerica
- **Matplotlib** - Geracao de graficos para analise quantitativa
- **pytest** - Framework de testes automatizados
- **Biblioteca padrao** - `random`, `collections.defaultdict`, `json`, `argparse`, `glob`

---

## Referencias

- Watkins, C. J. C. H., & Dayan, P. (1992). *Q-Learning*. Machine Learning, 8(3-4), 279-292.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Ammann, P., & Offutt, J. (2016). *Introduction to Software Testing* (2nd ed.). Cambridge University Press.

---

## Licenca

Este projeto e distribuido para fins educacionais e academicos.