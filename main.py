"""
Exemplo de Q-Learning aplicado a uma Máquina de Estados Finitos (FSM).

Este script cria uma FSM genérica com 6 estados e treina um agente
Q-Learning para encontrar o caminho ótimo de qualquer estado até
o estado-objetivo.

Estrutura da FSM:
                                  
    A ──(ir_B)──→ B ──(ir_D)──→ D    
    │             │              │    
  (ir_C)       (ir_C)        (ir_F)  
    ↓             ↓              ↓    
    C ──(ir_E)──→ E ──(ir_F)──→ F ★  
                                      
    ★ = Estado-objetivo (F)
    Recompensa: +100 ao chegar em F, -1 por cada passo
"""

from fsm import FSM
from q_learning import QLearningAgent


def create_example_fsm():
    """
    Cria uma FSM de exemplo com 6 estados (A-F).

    O agente deve aprender que:
    - De A, o melhor caminho é A→B→D→F (3 passos)
    - De B, o melhor caminho é B→D→F (2 passos)
    - De C, o melhor caminho é C→E→F (2 passos)
    - De D, o melhor caminho é D→F (1 passo)
    - De E, o melhor caminho é E→F (1 passo)
    """
    states = ["A", "B", "C", "D", "E", "F"]

    actions = ["ir_B", "ir_C", "ir_D", "ir_E", "ir_F"]

    # Definição das transições: (estado, ação) → próximo_estado
    transitions = {
        ("A", "ir_B"): "B",   # A → B
        ("A", "ir_C"): "C",   # A → C
        ("B", "ir_C"): "C",   # B → C (caminho mais longo)
        ("B", "ir_D"): "D",   # B → D
        ("C", "ir_E"): "E",   # C → E
        ("D", "ir_F"): "F",   # D → F (objetivo!)
        ("E", "ir_F"): "F",   # E → F (objetivo!)
    }

    # Recompensas: grande recompensa ao atingir o objetivo
    rewards = {
        ("A", "ir_B"): -1,
        ("A", "ir_C"): -1,
        ("B", "ir_C"): -1,
        ("B", "ir_D"): -1,
        ("C", "ir_E"): -1,
        ("D", "ir_F"): 100,   # Chegar ao objetivo!
        ("E", "ir_F"): 100,   # Chegar ao objetivo!
    }

    goal_states = {"F"}

    return FSM(
        states=states,
        actions=actions,
        transitions=transitions,
        rewards=rewards,
        goal_states=goal_states,
    )


def main():
    print("=" * 60)
    print("  Q-LEARNING PARA MAQUINA DE ESTADOS FINITOS")
    print("=" * 60)

    # --- 1. Criar a FSM ---
    print("\n[+] Criando FSM de exemplo...")
    fsm = create_example_fsm()
    print(f"   {fsm}")

    print("\n   Estrutura da FSM:")
    print("   A --(ir_B)--> B --(ir_D)--> D")
    print("   |              |              |")
    print("   (ir_C)       (ir_C)        (ir_F)")
    print("   v              v              v")
    print("   C --(ir_E)--> E --(ir_F)--> F *")
    print("   * = Estado-objetivo")

    # --- 2. Criar e treinar o agente ---
    print("\n[+] Criando agente Q-Learning...")
    agent = QLearningAgent(
        alpha=0.1,          # Taxa de aprendizado
        gamma=0.9,          # Fator de desconto
        epsilon=1.0,        # Exploracaoo inicial (100%)
        epsilon_decay=0.995, # Decaimento do epsilon
        epsilon_min=0.01,   # Epsilon minimo
    )

    print("\n[+] Iniciando treinamento (1000 episodios)...\n")
    metrics = agent.train(fsm, episodes=1000, max_steps_per_episode=50)

    print(f"\n[+] Metricas finais:")
    print(f"   Episodios: {metrics['total_episodes']}")
    print(f"   Epsilon final: {metrics['final_epsilon']:.4f}")
    print(f"   Recompensa media (ultimos 100): {metrics['avg_reward_last_100']:.1f}")
    print(f"   Passos medios (ultimos 100): {metrics['avg_steps_last_100']:.1f}")

    # --- 3. Mostrar Q-table ---
    agent.print_q_table(fsm)

    # --- 4. Mostrar caminhos otimos ---
    print("\n" + "=" * 60)
    print("  CAMINHOS OTIMOS APRENDIDOS")
    print("=" * 60)

    non_terminal_states = [s for s in fsm.states if not fsm.is_terminal(s)]

    for start in non_terminal_states:
        path = agent.get_optimal_path(fsm, start)

        # Formata o caminho como string
        path_str = ""
        for i, (state, action) in enumerate(path):
            if action is not None:
                path_str += f"{state} --({action})--> "
            else:
                if state in fsm.goal_states:
                    path_str += f"{state} *"
                else:
                    path_str += f"{state} (sem saida)"

        steps = len(path) - 1
        print(f"\n  De '{start}' ({steps} passo{'s' if steps != 1 else ''}):")
        print(f"    {path_str}")

    print("\n" + "=" * 60)
    print("  Treinamento concluido com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
