"""
Módulo que implementa o agente Q-Learning para navegação em FSM.

O Q-Learning é um algoritmo de aprendizado por reforço que aprende uma
política ótima sem necessitar de um modelo do ambiente. Ele mantém uma
tabela Q(s, a) que estima o valor esperado de tomar a ação 'a' no estado 's'.

Regra de atualização:
    Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

Onde:
    - α (alpha): taxa de aprendizado
    - γ (gamma): fator de desconto
    - r: recompensa recebida
    - s': próximo estado
"""

import random
from collections import defaultdict


class QLearningAgent:
    """Agente Q-Learning para navegação em Máquinas de Estados Finitos."""

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Inicializa o agente Q-Learning.

        Args:
            alpha: Taxa de aprendizado (0 < α ≤ 1).
                   Valores maiores fazem o agente aprender mais rápido,
                   mas podem causar instabilidade.
            gamma: Fator de desconto (0 ≤ γ < 1).
                   Valores próximos de 1 fazem o agente valorizar mais
                   recompensas futuras.
            epsilon: Taxa de exploração inicial (0 ≤ ε ≤ 1).
                     Probabilidade de escolher uma ação aleatória.
            epsilon_decay: Fator de decaimento do epsilon por episódio.
            epsilon_min: Valor mínimo do epsilon.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: mapeia (estado, ação) → valor Q
        self.q_table = defaultdict(float)

        # Métricas de treinamento
        self.rewards_per_episode = []
        self.steps_per_episode = []

        # Métricas de cobertura por episódio
        self.states_visited_per_episode = []
        self.transitions_visited_per_episode = []
        self.cumulative_states = set()
        self.cumulative_transitions = set()

    def choose_action(self, state, valid_actions):
        """
        Escolhe uma ação usando a política ε-greedy.

        Com probabilidade ε, escolhe uma ação aleatória (exploração).
        Com probabilidade (1 - ε), escolhe a melhor ação conhecida (exploitação).

        Args:
            state: O estado atual.
            valid_actions: Lista de ações válidas no estado atual.

        Returns:
            A ação escolhida.
        """
        if not valid_actions:
            return None

        # Exploração: ação aleatória
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploitação: melhor ação conhecida
        q_values = {action: self.q_table[(state, action)] for action in valid_actions}
        max_q = max(q_values.values())

        # Se houver empate, escolhe aleatoriamente entre as melhores
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_valid_actions):
        """
        Atualiza o valor Q usando a regra de atualização do Q-Learning.

        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

        Args:
            state: Estado atual (s).
            action: Ação tomada (a).
            reward: Recompensa recebida (r).
            next_state: Próximo estado (s').
            next_valid_actions: Ações válidas no próximo estado.
        """
        # Valor Q atual
        current_q = self.q_table[(state, action)]

        # Valor Q máximo do próximo estado
        if next_valid_actions:
            max_next_q = max(
                self.q_table[(next_state, a)] for a in next_valid_actions
            )
        else:
            max_next_q = 0.0  # Estado terminal

        # Regra de atualização
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        self.q_table[(state, action)] = current_q + self.alpha * td_error

    def decay_epsilon(self):
        """Aplica o decaimento do epsilon após cada episódio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, fsm, episodes=1000, max_steps_per_episode=100, verbose=True):
        """
        Treina o agente na FSM fornecida.

        Em cada episódio, o agente começa em um estado aleatório (não-terminal)
        e tenta chegar a um estado-objetivo.

        Args:
            fsm: Instância da FSM para treinar.
            episodes: Número de episódios de treinamento.
            max_steps_per_episode: Limite de passos por episódio.
            verbose: Se True, imprime progresso a cada 10% dos episódios.

        Returns:
            Dicionário com métricas de treinamento.
        """
        # Estados não-terminais para início dos episódios
        start_states = [s for s in fsm.states if not fsm.is_terminal(s)]

        if not start_states:
            raise ValueError("A FSM não possui estados não-terminais para iniciar o treinamento.")

        print_interval = max(1, episodes // 10)

        for episode in range(episodes):
            # Escolhe um estado inicial aleatório
            state = random.choice(start_states)
            total_reward = 0.0
            steps = 0
            episode_states = {state}
            episode_transitions = set()

            for step in range(max_steps_per_episode):
                valid_actions = fsm.get_valid_actions(state)

                if not valid_actions:
                    break  # Estado sem saída

                # Escolhe e executa ação
                action = self.choose_action(state, valid_actions)
                next_state, reward, done = fsm.step(state, action)

                # Registra cobertura
                episode_states.add(next_state)
                episode_transitions.add((state, action, next_state))

                # Obtém ações válidas do próximo estado
                next_valid_actions = fsm.get_valid_actions(next_state) if not done else []

                # Atualiza Q-table
                self.update(state, action, reward, next_state, next_valid_actions)

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

            # Decai epsilon
            self.decay_epsilon()

            # Imprime progresso
            if verbose and (episode + 1) % print_interval == 0:
                avg_reward = sum(self.rewards_per_episode[-print_interval:]) / print_interval
                avg_steps = sum(self.steps_per_episode[-print_interval:]) / print_interval
                print(
                    f"  Episodio {episode + 1:>5}/{episodes} | "
                    f"epsilon = {self.epsilon:.4f} | "
                    f"Recompensa media = {avg_reward:>7.1f} | "
                    f"Passos medios = {avg_steps:.1f}"
                )

        return {
            "total_episodes": episodes,
            "final_epsilon": self.epsilon,
            "avg_reward_last_100": sum(self.rewards_per_episode[-100:]) / min(100, len(self.rewards_per_episode)),
            "avg_steps_last_100": sum(self.steps_per_episode[-100:]) / min(100, len(self.steps_per_episode)),
        }

    def get_optimal_path(self, fsm, start_state, max_steps=50):
        """
        Extrai o caminho ótimo aprendido a partir de um estado.

        Usa apenas a política greedy (sem exploração) para
        seguir as melhores ações aprendidas.

        Args:
            fsm: Instância da FSM.
            start_state: Estado de início.
            max_steps: Limite de passos para evitar loops infinitos.

        Returns:
            Lista de tuplas (estado, ação) representando o caminho,
            terminando com (estado_final, None).
        """
        path = []
        state = start_state

        for _ in range(max_steps):
            if fsm.is_terminal(state):
                path.append((state, None))
                break

            valid_actions = fsm.get_valid_actions(state)
            if not valid_actions:
                path.append((state, None))
                break

            # Escolhe a melhor ação (greedy)
            q_values = {a: self.q_table[(state, a)] for a in valid_actions}
            best_action = max(q_values, key=q_values.get)

            path.append((state, best_action))
            next_state, _, done = fsm.step(state, best_action)
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
            threshold: Fração de cobertura desejada (0.0 a 1.0). Padrão: 1.0 (100%).

        Returns:
            Número do episódio (1-indexed) ou None se não atingiu.
        """
        for i, count in enumerate(self.states_visited_per_episode):
            if count / total_states >= threshold:
                return i + 1
        return None

    def print_q_table(self, fsm):
        """
        Imprime a Q-table de forma organizada.

        Args:
            fsm: Instância da FSM (para iterar sobre estados e ações).
        """
        print("\n" + "=" * 60)
        print("  Q-TABLE")
        print("=" * 60)

        for state in fsm.states:
            valid_actions = fsm.get_valid_actions(state)
            if not valid_actions:
                continue

            print(f"\n  Estado '{state}':")
            for action in valid_actions:
                q_val = self.q_table[(state, action)]
                indicator = " <-- melhor" if q_val == max(
                    self.q_table[(state, a)] for a in valid_actions
                ) else ""
                print(f"    {action:>12}: Q = {q_val:>8.2f}{indicator}")

        print("\n" + "=" * 60)
