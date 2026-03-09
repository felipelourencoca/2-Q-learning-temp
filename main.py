"""
Q-Learning aplicado a uma Máquina de Estados Finitos (FSM).

Este script carrega uma FSM a partir de um arquivo JSON externo
e treina um agente Q-Learning para explorar os estados e transições.

Uso:
    python main.py --json caminho/para/arquivo.json
    python main.py --json caminho/para/arquivo.json --goal estado_objetivo
    python main.py --json caminho/para/arquivo.json --goal estado_objetivo --episodes 2000
"""

import argparse

from fsm_loader import load_fsm_from_json
from q_learning import QLearningAgent


def parse_args():
    """Processa argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Q-Learning para Máquina de Estados Finitos (FSM)"
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Caminho para o arquivo JSON contendo a definição da FSM.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        nargs="*",
        help="Estado(s)-objetivo da FSM (separados por espaço).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Número de episódios de treinamento (padrão: 1000).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Máximo de passos por episódio (padrão: 50).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Taxa de aprendizado (padrão: 0.1).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Fator de desconto (padrão: 0.9).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Taxa de exploração inicial (padrão: 1.0).",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Fator de decaimento do epsilon (padrão: 0.995).",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.01,
        help="Valor mínimo do epsilon (padrão: 0.01).",
    )
    return parser.parse_args()


def print_fsm_info(fsm):
    """Imprime informações sobre a FSM carregada."""
    print(f"   {fsm}")
    print(f"   Estado inicial: {fsm.initial_state or 'N/A'}")
    print(f"\n   Transicoes:")
    for (state, action), target in sorted(fsm.transitions.items()):
        reward = fsm.rewards.get((state, action), fsm.default_reward)
        goal_marker = " *" if target in fsm.goal_states else ""
        print(f"     {state} --({action})--> {target}{goal_marker}  [r={reward}]")
    if fsm.goal_states:
        print(f"\n   Estados-objetivo: {fsm.goal_states}")
    else:
        print(f"\n   Sem estados-objetivo definidos (exploracao livre).")


def main():
    args = parse_args()

    print("=" * 60)
    print("  Q-LEARNING PARA MAQUINA DE ESTADOS FINITOS")
    print("=" * 60)

    # --- 1. Carregar a FSM do arquivo JSON ---
    print(f"\n[+] Carregando FSM do arquivo: {args.json}")
    goal_states = set(args.goal) if args.goal else None
    fsm = load_fsm_from_json(args.json, goal_states=goal_states)
    print_fsm_info(fsm)

    # --- 2. Criar e treinar o agente ---
    print("\n[+] Criando agente Q-Learning...")
    agent = QLearningAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
    )
    print(f"   alpha={args.alpha}, gamma={args.gamma}, epsilon={args.epsilon}, "
          f"epsilon_decay={args.epsilon_decay}, epsilon_min={args.epsilon_min}")

    episodes = args.episodes
    max_steps = args.max_steps
    print(f"\n[+] Iniciando treinamento ({episodes} episodios, max {max_steps} passos/episodio)...\n")
    metrics = agent.train(fsm, episodes=episodes, max_steps_per_episode=max_steps)

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
