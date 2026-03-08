"""
Módulo para carregar uma FSM a partir de um arquivo JSON externo.

Formato JSON esperado:
{
    "initial": "nome_do_estado_inicial",
    "states": [
        {
            "state": "nome_do_estado",
            "transitions": [
                {
                    "input": "nome_da_acao",
                    "output": ["efeito1", "efeito2"],
                    "target": "nome_do_estado_destino"
                }
            ]
        }
    ]
}
"""

import json
from fsm import FSM


def load_fsm_from_json(filepath, goal_states=None, goal_reward=100, step_reward=-1):
    """
    Carrega uma FSM a partir de um arquivo JSON.

    Lê o arquivo JSON, extrai estados, ações e transições, e cria
    uma instância da classe FSM.

    Args:
        filepath: Caminho para o arquivo JSON.
        goal_states: Conjunto opcional de estados-objetivo.
                     Se None, a FSM não terá estados terminais.
        goal_reward: Recompensa ao atingir um estado-objetivo (padrão: 100).
        step_reward: Recompensa padrão para cada transição (padrão: -1).

    Returns:
        Instância de FSM configurada a partir do JSON.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    initial_state = data.get("initial")
    states_data = data.get("states", [])

    # Extrair estados, ações e transições
    states = []
    actions_set = set()
    transitions = {}
    rewards = {}

    for state_obj in states_data:
        state_name = state_obj["state"]
        states.append(state_name)

        for transition in state_obj.get("transitions", []):
            action = transition["input"]
            target = transition["target"]

            actions_set.add(action)
            transitions[(state_name, action)] = target

            # Atribuir recompensa
            if goal_states and target in goal_states:
                rewards[(state_name, action)] = goal_reward
            else:
                rewards[(state_name, action)] = step_reward

    actions = sorted(actions_set)

    if goal_states is None:
        goal_states = set()

    fsm = FSM(
        states=states,
        actions=actions,
        transitions=transitions,
        rewards=rewards,
        goal_states=goal_states,
        initial_state=initial_state,
    )

    return fsm
