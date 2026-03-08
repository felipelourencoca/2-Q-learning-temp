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
"""

import pytest
from fsm import FSM
from q_learning import QLearningAgent


# ===========================================================================
#  FIXTURE: FSM de exemplo (mesma do main.py)
# ===========================================================================

@pytest.fixture
def sample_fsm():
    """
    Cria a FSM de exemplo com 6 estados:

        A --(ir_B)--> B --(ir_D)--> D
        |             |              |
      (ir_C)       (ir_C)        (ir_F)
        ↓             ↓              ↓
        C --(ir_E)--> E --(ir_F)--> F ★

    Estados: {A, B, C, D, E, F}
    Estado-objetivo: F
    """
    states = ["A", "B", "C", "D", "E", "F"]
    actions = ["ir_B", "ir_C", "ir_D", "ir_E", "ir_F"]
    transitions = {
        ("A", "ir_B"): "B",
        ("A", "ir_C"): "C",
        ("B", "ir_C"): "C",
        ("B", "ir_D"): "D",
        ("C", "ir_E"): "E",
        ("D", "ir_F"): "F",
        ("E", "ir_F"): "F",
    }
    rewards = {
        ("A", "ir_B"): -1,
        ("A", "ir_C"): -1,
        ("B", "ir_C"): -1,
        ("B", "ir_D"): -1,
        ("C", "ir_E"): -1,
        ("D", "ir_F"): 100,
        ("E", "ir_F"): 100,
    }
    goal_states = {"F"}
    return FSM(states, actions, transitions, rewards, goal_states)


@pytest.fixture
def trained_agent(sample_fsm):
    """Retorna um agente treinado na FSM de exemplo."""
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

    Cada teste exercita um caminho que VISITA um ou mais estados da FSM.
    O conjunto completo de testes deve cobrir TODOS os estados S = {A,B,C,D,E,F}.

    Mapeamento de cobertura:
    ┌──────────────────────────────┬────────────────────────┐
    │ Caso de teste                │ Estados cobertos       │
    ├──────────────────────────────┼────────────────────────┤
    │ test_state_A_reachable       │ A                      │
    │ test_state_B_via_A           │ A, B                   │
    │ test_state_C_via_A           │ A, C                   │
    │ test_state_D_via_A_B         │ A, B, D                │
    │ test_state_E_via_A_C         │ A, C, E                │
    │ test_state_F_via_D           │ D, F                   │
    │ test_state_F_via_E           │ E, F                   │
    │ test_all_states_covered      │ A, B, C, D, E, F (all) │
    └──────────────────────────────┴────────────────────────┘
    """

    # --- Cobertura individual de cada estado ---

    def test_state_A_reachable(self, sample_fsm):
        """
        State Coverage: Estado A é alcançável (estado inicial).

        Verifica que A existe na FSM, é não-terminal, e possui
        ações válidas de saída.
        """
        assert "A" in sample_fsm.states, "Estado A deve existir na FSM"
        assert not sample_fsm.is_terminal("A"), "Estado A não deve ser terminal"
        valid = sample_fsm.get_valid_actions("A")
        assert len(valid) > 0, "Estado A deve ter ações de saída"

    def test_state_B_via_A(self, sample_fsm):
        """
        State Coverage: Estado B é alcançável a partir de A.

        Caminho: A --(ir_B)--> B
        Estados visitados: {A, B}
        """
        next_state, reward, done = sample_fsm.step("A", "ir_B")
        assert next_state == "B", "Transição A--(ir_B) deve levar ao estado B"
        assert not done, "Estado B não é terminal"
        assert reward == -1, "Recompensa de A→B deve ser -1"

    def test_state_C_via_A(self, sample_fsm):
        """
        State Coverage: Estado C é alcançável a partir de A.

        Caminho: A --(ir_C)--> C
        Estados visitados: {A, C}
        """
        next_state, reward, done = sample_fsm.step("A", "ir_C")
        assert next_state == "C", "Transição A--(ir_C) deve levar ao estado C"
        assert not done, "Estado C não é terminal"

    def test_state_C_via_B(self, sample_fsm):
        """
        State Coverage: Estado C é alcançável via caminho alternativo B.

        Caminho: B --(ir_C)--> C
        Estados visitados: {B, C}
        """
        next_state, _, done = sample_fsm.step("B", "ir_C")
        assert next_state == "C", "Transição B--(ir_C) deve levar ao estado C"
        assert not done, "Estado C não é terminal"

    def test_state_D_via_A_B(self, sample_fsm):
        """
        State Coverage: Estado D é alcançável via caminho A → B → D.

        Caminho: A --(ir_B)--> B --(ir_D)--> D
        Estados visitados: {A, B, D}
        """
        s1, _, _ = sample_fsm.step("A", "ir_B")
        assert s1 == "B"
        s2, _, done = sample_fsm.step(s1, "ir_D")
        assert s2 == "D", "Transição B--(ir_D) deve levar ao estado D"
        assert not done, "Estado D não é terminal"

    def test_state_E_via_A_C(self, sample_fsm):
        """
        State Coverage: Estado E é alcançável via caminho A → C → E.

        Caminho: A --(ir_C)--> C --(ir_E)--> E
        Estados visitados: {A, C, E}
        """
        s1, _, _ = sample_fsm.step("A", "ir_C")
        assert s1 == "C"
        s2, _, done = sample_fsm.step(s1, "ir_E")
        assert s2 == "E", "Transição C--(ir_E) deve levar ao estado E"
        assert not done, "Estado E não é terminal"

    def test_state_F_via_D(self, sample_fsm):
        """
        State Coverage: Estado F (goal) é alcançável via D.

        Caminho: D --(ir_F)--> F ★
        Estados visitados: {D, F}
        """
        next_state, reward, done = sample_fsm.step("D", "ir_F")
        assert next_state == "F", "Transição D--(ir_F) deve levar ao estado F"
        assert done, "Estado F DEVE ser terminal (goal state)"
        assert reward == 100, "Recompensa de D→F deve ser +100"

    def test_state_F_via_E(self, sample_fsm):
        """
        State Coverage: Estado F (goal) é alcançável via E.

        Caminho: E --(ir_F)--> F ★
        Estados visitados: {E, F}
        """
        next_state, reward, done = sample_fsm.step("E", "ir_F")
        assert next_state == "F", "Transição E--(ir_F) deve levar ao estado F"
        assert done, "Estado F DEVE ser terminal (goal state)"
        assert reward == 100, "Recompensa de E→F deve ser +100"

    # --- Verificação formal do critério de State Coverage ---

    def test_all_states_covered(self, sample_fsm):
        """
        VERIFICAÇÃO FORMAL DO CRITÉRIO DE STATE COVERAGE.

        Percorre TODOS os caminhos possíveis na FSM e verifica que
        todos os estados são alcançados. Este teste garante que a
        suíte satisfaz o critério:

            |Estados visitados| / |Total de estados| = 100%
        """
        all_states = set(sample_fsm.states)
        visited_states = set()

        # Caminho 1: A → B → D → F (cobre A, B, D, F)
        visited_states.add("A")
        s, _, _ = sample_fsm.step("A", "ir_B")
        visited_states.add(s)  # B
        s, _, _ = sample_fsm.step(s, "ir_D")
        visited_states.add(s)  # D
        s, _, _ = sample_fsm.step(s, "ir_F")
        visited_states.add(s)  # F

        # Caminho 2: A → C → E → F (cobre C, E)
        visited_states.add("A")
        s, _, _ = sample_fsm.step("A", "ir_C")
        visited_states.add(s)  # C
        s, _, _ = sample_fsm.step(s, "ir_E")
        visited_states.add(s)  # E
        s, _, _ = sample_fsm.step(s, "ir_F")
        visited_states.add(s)  # F

        # Verificação: todos os estados foram cobertos?
        uncovered = all_states - visited_states
        coverage = len(visited_states) / len(all_states) * 100

        assert uncovered == set(), (
            f"State Coverage FALHOU! Estados não cobertos: {uncovered}. "
            f"Cobertura: {coverage:.0f}%"
        )
        assert coverage == 100.0, f"State Coverage deve ser 100%, obteve {coverage:.0f}%"


# ===========================================================================
#  STATE COVERAGE com Q-LEARNING (pós-treinamento)
# ===========================================================================

class TestStateCoverageQLearning:
    """
    Verifica que o agente Q-Learning treinado cobre todos os estados
    quando calculamos os caminhos ótimos a partir de cada estado.

    Este teste valida que o Q-Learning aprendeu caminhos que, em
    conjunto, cobrem toda a FSM.
    """

    def test_qlearning_covers_all_states(self, sample_fsm, trained_agent):
        """
        Verificação de State Coverage nos caminhos ótimos do Q-Learning.

        Para cada estado não-terminal, extrai o caminho ótimo e
        registra todos os estados visitados. O critério de State
        Coverage é satisfeito se a união de todos os caminhos
        cobre S = {A, B, C, D, E, F}.
        """
        all_states = set(sample_fsm.states)
        visited_states = set()

        non_terminal = [s for s in sample_fsm.states if not sample_fsm.is_terminal(s)]

        for start in non_terminal:
            path = trained_agent.get_optimal_path(sample_fsm, start)
            for state, _ in path:
                visited_states.add(state)

        uncovered = all_states - visited_states
        coverage = len(visited_states) / len(all_states) * 100

        assert uncovered == set(), (
            f"State Coverage do Q-Learning FALHOU! "
            f"Estados não cobertos: {uncovered}. Cobertura: {coverage:.0f}%"
        )

    def test_qlearning_optimal_paths_reach_goal(self, sample_fsm, trained_agent):
        """
        Verifica que todos os caminhos ótimos terminam no estado-objetivo F.

        Critério complementar ao State Coverage: não basta visitar
        todos os estados, os caminhos devem atingir o objetivo.
        """
        non_terminal = [s for s in sample_fsm.states if not sample_fsm.is_terminal(s)]

        for start in non_terminal:
            path = trained_agent.get_optimal_path(sample_fsm, start)
            final_state = path[-1][0]
            assert final_state in sample_fsm.goal_states, (
                f"Caminho ótimo de '{start}' terminou em '{final_state}' "
                f"ao invés de um estado-objetivo {sample_fsm.goal_states}"
            )

    def test_qlearning_state_visit_count(self, sample_fsm, trained_agent):
        """
        Relatório quantitativo: conta quantas vezes cada estado é
        visitado nos caminhos ótimos.

        Isso demonstra a cobertura de cada estado individualmente.
        """
        state_visit_count = {s: 0 for s in sample_fsm.states}
        non_terminal = [s for s in sample_fsm.states if not sample_fsm.is_terminal(s)]

        for start in non_terminal:
            path = trained_agent.get_optimal_path(sample_fsm, start)
            for state, _ in path:
                state_visit_count[state] += 1

        # Todos os estados devem ter sido visitados ao menos uma vez
        for state, count in state_visit_count.items():
            assert count >= 1, (
                f"Estado '{state}' não foi visitado em nenhum caminho ótimo "
                f"(State Coverage não satisfeito para este estado)"
            )


# ===========================================================================
#  RELATÓRIO DE STATE COVERAGE
# ===========================================================================

class TestStateCoverageReport:
    """
    Gera um relatório detalhado da cobertura de estados.

    Este relatório mostra:
    - Porcentagem de cobertura
    - Quais estados foram cobertos
    - Quais caminhos cobrem cada estado
    """

    def test_generate_coverage_report(self, sample_fsm, trained_agent, capsys):
        """
        Imprime um relatório de State Coverage no output do pytest.

        Execute com: pytest test_state_coverage.py -v -s
        """
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
        print(f"\n  Total de estados:    {len(all_states)}")
        print(f"  Estados cobertos:    {len(visited_states)}")
        print(f"  Cobertura:           {coverage:.0f}%")
        print(f"\n  {'Estado':<10} {'Coberto':<10} {'Visitas'}")
        print(f"  {'-'*10} {'-'*10} {'-'*10}")

        for state in sorted(sample_fsm.states):
            covered = "[x]" if state in visited_states else "[ ]"
            visit_count = len(state_paths[state])
            print(f"  {state:<10} {covered:<10} {visit_count}")

        print(f"\n  Caminhos que cobrem cada estado:")
        for state in sorted(sample_fsm.states):
            if state_paths[state]:
                print(f"\n  Estado '{state}':")
                for p in state_paths[state]:
                    print(f"  {p}")

        print("\n" + "=" * 60)

        # Assertion final
        assert coverage == 100.0, f"State Coverage incompleto: {coverage:.0f}%"
