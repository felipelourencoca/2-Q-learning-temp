"""
Módulo que define a Máquina de Estados Finitos (FSM) genérica.

A FSM é composta por:
- Um conjunto de estados
- Um conjunto de ações
- Uma função de transição: (estado, ação) → próximo_estado
- Uma função de recompensa: (estado, ação) → recompensa
- Um conjunto de estados-objetivo (terminais)
"""


class FSM:
    """Máquina de Estados Finitos genérica para uso com Q-Learning."""

    def __init__(self, states, actions, transitions, rewards, goal_states, default_reward=-1.0):
        """
        Inicializa a FSM.

        Args:
            states: Lista de estados (ex: ["A", "B", "C"]).
            actions: Lista de todas as ações possíveis (ex: ["ir_B", "ir_C"]).
            transitions: Dicionário de transições {(estado, ação): próximo_estado}.
            rewards: Dicionário de recompensas {(estado, ação): recompensa}.
                     Se uma transição não tiver recompensa definida, usa default_reward.
            goal_states: Conjunto de estados terminais/objetivo.
            default_reward: Recompensa padrão para transições sem recompensa definida.
        """
        self.states = list(states)
        self.actions = list(actions)
        self.transitions = dict(transitions)
        self.rewards = dict(rewards)
        self.goal_states = set(goal_states)
        self.default_reward = default_reward

        # Pré-calcular ações válidas por estado para acesso rápido
        self._valid_actions = {}
        for state in self.states:
            self._valid_actions[state] = [
                action for action in self.actions
                if (state, action) in self.transitions
            ]

    def get_valid_actions(self, state):
        """
        Retorna a lista de ações válidas a partir de um estado.

        Args:
            state: O estado atual.

        Returns:
            Lista de ações que possuem transição definida a partir do estado.
        """
        return self._valid_actions.get(state, [])

    def step(self, state, action):
        """
        Executa uma ação a partir de um estado.

        Args:
            state: O estado atual.
            action: A ação a ser executada.

        Returns:
            Tupla (próximo_estado, recompensa, terminado):
                - próximo_estado: o estado resultante da transição
                - recompensa: o valor da recompensa recebida
                - terminado: True se o próximo estado é um estado-objetivo

        Raises:
            ValueError: Se a transição (estado, ação) não é válida.
        """
        if (state, action) not in self.transitions:
            raise ValueError(
                f"Transição inválida: estado='{state}', ação='{action}'. "
                f"Ações válidas: {self.get_valid_actions(state)}"
            )

        next_state = self.transitions[(state, action)]
        reward = self.rewards.get((state, action), self.default_reward)
        done = next_state in self.goal_states

        return next_state, reward, done

    def is_terminal(self, state):
        """Verifica se um estado é terminal (estado-objetivo)."""
        return state in self.goal_states

    def __repr__(self):
        return (
            f"FSM(estados={self.states}, "
            f"acoes={self.actions}, "
            f"transicoes={len(self.transitions)}, "
            f"objetivos={self.goal_states})"
        )
