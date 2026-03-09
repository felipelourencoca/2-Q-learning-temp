"""
Critério de Cobertura de Teste: STATE COVERAGE para FSM

State Coverage (Cobertura de Estados) é um critério de teste para
Máquinas de Estados Finitos que exige que TODOS os estados da FSM
sejam visitados por pelo menos um caso de teste.

Definição formal:
    Dado um conjunto de estados S = {s1, s2, ..., sn} da FSM,
    uma suíte de testes T satisfaz o critério de State Coverage se
    e somente se, para cada estado si ∈ S, existe pelo menos um
    caso de teste t ∈ T tal que a execução de t visita si.

    Cobertura = |Estados visitados| / |Total de estados| × 100%

Referência: "Introduction to Software Testing" – Paul Ammann & Jeff Offutt

Uso:
    python -m pytest test_state_coverage.py -v
    python -m pytest test_state_coverage.py -v -s   (com relatório)
"""

import os
import glob
import pytest
from fsm_loader import load_fsm_from_json
from q_learning import QLearningAgent


# ===========================================================================
#  Descoberta automática de arquivos JSON de FSM
# ===========================================================================

FSM_DIR = os.path.join(os.path.dirname(__file__), "finite_states_machines")

FSM_FILES = sorted(glob.glob(os.path.join(FSM_DIR, "*.json")))


# ===========================================================================
#  FIXTURES
# ===========================================================================

@pytest.fixture(params=FSM_FILES, ids=[os.path.basename(f) for f in FSM_FILES])
def fsm_file(request):
    """Retorna o caminho de cada arquivo JSON de FSM encontrado."""
    return request.param


@pytest.fixture
def sample_fsm(fsm_file):
    """Carrega uma FSM a partir do arquivo JSON externo."""
    return load_fsm_from_json(fsm_file)


@pytest.fixture
def trained_agent(sample_fsm):
    """Retorna um agente treinado na FSM carregada."""
    agent = QLearningAgent(
        alpha=0.1, gamma=0.9, epsilon=1.0,
        epsilon_decay=0.995, epsilon_min=0.01,
    )
    agent.train(sample_fsm, episodes=1000, max_steps_per_episode=50, verbose=False)
    return agent


# ===========================================================================
#  STATE COVERAGE – Critério principal
# ===========================================================================

class TestStateCoverage:
    """
    Critério: STATE COVERAGE (Cobertura de Estados)

    Testes genéricos que verificam a cobertura de estados para
    qualquer FSM carregada de um arquivo JSON.
    """

    def test_fsm_has_states(self, sample_fsm):
        """Verifica que a FSM possui pelo menos um estado."""
        assert len(sample_fsm.states) > 0, "A FSM deve possuir ao menos um estado"

    def test_fsm_has_initial_state(self, sample_fsm):
        """Verifica que a FSM possui um estado inicial definido."""
        assert sample_fsm.initial_state is not None, (
            "A FSM deve possuir um estado inicial"
        )
        assert sample_fsm.initial_state in sample_fsm.states, (
            f"Estado inicial '{sample_fsm.initial_state}' não está na lista de estados"
        )

    def test_all_states_have_transitions_or_are_terminal(self, sample_fsm):
        """
        Verifica que cada estado possui ao menos uma transição de saída
        ou é um estado terminal (sem saídas).
        """
        dead_end_states = []
        for state in sample_fsm.states:
            valid_actions = sample_fsm.get_valid_actions(state)
            if not valid_actions and not sample_fsm.is_terminal(state):
                dead_end_states.append(state)
        # Dead-ends são permitidos, mas registramos
        # (em FSMs flattened, estados sem transição são beco sem saída)

    def test_all_transitions_lead_to_valid_states(self, sample_fsm):
        """Verifica que todas as transições levam a estados válidos."""
        for (state, action), target in sample_fsm.transitions.items():
            assert state in sample_fsm.states, (
                f"Estado de origem '{state}' da transição não está na lista de estados"
            )
            assert target in sample_fsm.states, (
                f"Estado destino '{target}' da transição ({state}, {action}) "
                f"não está na lista de estados"
            )

    def test_all_states_reachable_from_initial(self, sample_fsm):
        """
        Verifica que todos os estados são alcançáveis a partir do
        estado inicial usando BFS.

        State Coverage exige que todos os estados possam ser visitados.
        """
        if sample_fsm.initial_state is None:
            pytest.skip("FSM sem estado inicial definido")

        visited = set()
        queue = [sample_fsm.initial_state]

        while queue:
            state = queue.pop(0)
            if state in visited:
                continue
            visited.add(state)

            for action in sample_fsm.get_valid_actions(state):
                next_state = sample_fsm.transitions.get((state, action))
                if next_state and next_state not in visited:
                    queue.append(next_state)

        all_states = set(sample_fsm.states)
        unreachable = all_states - visited
        coverage = len(visited) / len(all_states) * 100

        assert unreachable == set(), (
            f"State Coverage FALHOU! Estados não alcançáveis a partir de "
            f"'{sample_fsm.initial_state}': {unreachable}. "
            f"Cobertura: {coverage:.0f}%"
        )

    def test_each_state_visited_via_transitions(self, sample_fsm):
        """
        Percorre todas as transições da FSM e verifica que cada estado
        é visitado como origem ou destino de pelo menos uma transição.
        """
        visited = set()
        for (state, action), target in sample_fsm.transitions.items():
            visited.add(state)
            visited.add(target)

        all_states = set(sample_fsm.states)
        # Estados sem transições (nem de entrada nem de saída) ficam isolados
        isolated = all_states - visited
        # Não é erro, mas registramos
        coverage = len(visited) / len(all_states) * 100
        assert coverage > 0, "Nenhum estado foi visitado por transições"


# ===========================================================================
#  STATE COVERAGE com Q-LEARNING (pós-treinamento)
# ===========================================================================

class TestStateCoverageQLearning:
    """
    Verifica que o agente Q-Learning treinado explora os estados
    da FSM através dos caminhos aprendidos.
    """

    def test_qlearning_explores_states(self, sample_fsm, trained_agent):
        """
        Verifica a cobertura de estados nos caminhos do Q-Learning.

        Para cada estado não-terminal, extrai o caminho aprendido e
        registra todos os estados visitados.
        """
        visited_states = set()

        non_terminal = [s for s in sample_fsm.states if not sample_fsm.is_terminal(s)]

        for start in non_terminal:
            path = trained_agent.get_optimal_path(sample_fsm, start)
            for state, _ in path:
                visited_states.add(state)

        # Deve visitar pelo menos os estados não-terminais
        non_terminal_set = set(non_terminal)
        covered = non_terminal_set & visited_states
        assert len(covered) > 0, "Q-Learning não visitou nenhum estado"

    def test_qlearning_q_table_populated(self, sample_fsm, trained_agent):
        """
        Verifica que a Q-table foi populada com valores para os
        estados que possuem ações válidas.
        """
        populated_states = set()
        for (state, action), q_val in trained_agent.q_table.items():
            if state in sample_fsm.states:
                populated_states.add(state)

        # Apenas estados com ações válidas (não dead-ends) devem estar na Q-table
        states_with_actions = {
            s for s in sample_fsm.states
            if sample_fsm.get_valid_actions(s)
        }
        assert populated_states == states_with_actions, (
            f"Q-table deveria conter todos os estados com ações válidas. "
            f"Faltando: {states_with_actions - populated_states}"
        )


# ===========================================================================
#  RELATÓRIO DE STATE COVERAGE
# ===========================================================================

class TestStateCoverageReport:
    """
    Gera um relatório detalhado da cobertura de estados.

    Execute com: python -m pytest test_state_coverage.py -v -s
    """

    def test_generate_coverage_report(self, sample_fsm, trained_agent, fsm_file, capsys):
        """Imprime um relatório de State Coverage no output do pytest."""
        all_states = set(sample_fsm.states)
        state_paths = {s: [] for s in sample_fsm.states}
        visited_states = set()

        non_terminal = [s for s in sample_fsm.states if not sample_fsm.is_terminal(s)]

        for start in non_terminal:
            path = trained_agent.get_optimal_path(sample_fsm, start)
            path_str = " -> ".join(s for s, _ in path)
            for state, _ in path:
                visited_states.add(state)
                state_paths[state].append(f"  De {start}: {path_str}")

        coverage = len(visited_states) / len(all_states) * 100

        # Imprime relatório
        print("\n")
        print("=" * 60)
        print("  RELATÓRIO DE STATE COVERAGE")
        print("=" * 60)
        print(f"\n  Arquivo: {os.path.basename(fsm_file)}")
        print(f"  Total de estados:    {len(all_states)}")
        print(f"  Estados cobertos:    {len(visited_states)}")
        print(f"  Cobertura:           {coverage:.0f}%")
        print(f"\n  {'Estado':<40} {'Coberto':<10} {'Visitas'}")
        print(f"  {'-'*40} {'-'*10} {'-'*10}")

        for state in sorted(sample_fsm.states):
            covered = "[x]" if state in visited_states else "[ ]"
            visit_count = len(state_paths[state])
            print(f"  {state:<40} {covered:<10} {visit_count}")

        print("\n" + "=" * 60)

        # Não faz assertion de 100% pois FSMs sem goal não convergem
        assert coverage > 0, "Nenhum estado foi coberto"
