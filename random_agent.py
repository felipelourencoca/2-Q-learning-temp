"""
Agente de Busca Aleatória (Baseline) para FSM.

O RandomAgent escolhe ações aleatórias sem nenhum aprendizado.
Serve como baseline para comparação com o Q-Learning,
demonstrando que o aprendizado por reforço traz vantagens
em relação à exploração puramente aleatória.
"""

import random


class RandomAgent:
    """Agente que escolhe ações aleatórias — baseline sem aprendizado."""

    def __init__(self):
        """Inicializa o agente aleatório."""
        # Métricas de treinamento (mesma interface do QLearningAgent)
        self.rewards_per_episode = []
        self.steps_per_episode = []

        # Métricas de cobertura por episódio
        self.states_visited_per_episode = []
        self.transitions_visited_per_episode = []
        self.cumulative_states = set()
        self.cumulative_transitions = set()

        # Sem Q-table — apenas para compatibilidade
        self.q_table = {}

    def choose_action(self, state, valid_actions):
        """
        Escolhe uma ação aleatória (sem política aprendida).

        Args:
            state: O estado atual (não utilizado).
            valid_actions: Lista de ações válidas no estado atual.

        Returns:
            Uma ação aleatória.
        """
        if not valid_actions:
            return None
        return random.choice(valid_actions)

    def train(self, fsm, episodes=1000, max_steps_per_episode=100, verbose=True):
        """
        Executa episódios de exploração aleatória na FSM.

        Args:
            fsm: Instância da FSM para explorar.
            episodes: Número de episódios.
            max_steps_per_episode: Limite de passos por episódio.
            verbose: Se True, imprime progresso.

        Returns:
            Dicionário com métricas de treinamento.
        """
        start_states = [s for s in fsm.states if not fsm.is_terminal(s)]

        if not start_states:
            raise ValueError("A FSM não possui estados não-terminais.")

        print_interval = max(1, episodes // 10)

        for episode in range(episodes):
            state = random.choice(start_states)
            total_reward = 0.0
            steps = 0
            episode_states = {state}
            episode_transitions = set()

            for step in range(max_steps_per_episode):
                valid_actions = fsm.get_valid_actions(state)

                if not valid_actions:
                    break

                action = self.choose_action(state, valid_actions)
                next_state, reward, done = fsm.step(state, action)

                episode_states.add(next_state)
                episode_transitions.add((state, action, next_state))

                total_reward += reward
                steps += 1
                state = next_state

                if done:
                    break

            # Acumula métricas de cobertura
            self.cumulative_states.update(episode_states)
            self.cumulative_transitions.update(episode_transitions)

            # Registra métricas
            self.rewards_per_episode.append(total_reward)
            self.steps_per_episode.append(steps)
            self.states_visited_per_episode.append(len(self.cumulative_states))
            self.transitions_visited_per_episode.append(len(self.cumulative_transitions))

            if verbose and (episode + 1) % print_interval == 0:
                avg_reward = sum(self.rewards_per_episode[-print_interval:]) / print_interval
                avg_steps = sum(self.steps_per_episode[-print_interval:]) / print_interval
                print(
                    f"  Episodio {episode + 1:>5}/{episodes} | "
                    f"Recompensa media = {avg_reward:>7.1f} | "
                    f"Passos medios = {avg_steps:.1f}"
                )

        return {
            "total_episodes": episodes,
            "avg_reward_last_100": sum(self.rewards_per_episode[-100:]) / min(100, len(self.rewards_per_episode)),
            "avg_steps_last_100": sum(self.steps_per_episode[-100:]) / min(100, len(self.steps_per_episode)),
        }

    def get_optimal_path(self, fsm, start_state, max_steps=50):
        """
        Gera um caminho aleatório a partir de um estado.

        Args:
            fsm: Instância da FSM.
            start_state: Estado de início.
            max_steps: Limite de passos.

        Returns:
            Lista de tuplas (estado, ação) representando o caminho.
        """
        path = []
        state = start_state
        visited = set()

        for _ in range(max_steps):
            if fsm.is_terminal(state):
                path.append((state, None))
                break

            valid_actions = fsm.get_valid_actions(state)
            if not valid_actions:
                path.append((state, None))
                break

            action = random.choice(valid_actions)
            path.append((state, action))

            next_state, _, done = fsm.step(state, action)

            # Evitar loops infinitos
            if next_state in visited:
                path.append((next_state, None))
                break
            visited.add(next_state)
            state = next_state

            if done:
                path.append((state, None))
                break

        return path

    def get_convergence_episode(self, total_states, threshold=1.0):
        """
        Retorna o episódio em que a cobertura de estados atingiu o threshold.

        Args:
            total_states: Número total de estados da FSM.
            threshold: Fração de cobertura desejada (0.0 a 1.0).

        Returns:
            Número do episódio (1-indexed) ou None se não atingiu.
        """
        for i, count in enumerate(self.states_visited_per_episode):
            if count / total_states >= threshold:
                return i + 1
        return None
