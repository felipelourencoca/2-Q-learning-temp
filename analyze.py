"""
Análise Quantitativa do Q-Learning para FSM — Gráficos para Dissertação.

Gera gráficos comparativos entre Q-Learning e Random Agent:
1. Curvas de aprendizado (recompensa × episódios)
2. Convergência da cobertura de estados e transições
3. Análise de sensibilidade de hiperparâmetros
4. Eficiência de caminho (Q-Learning vs BFS ótimo)

Uso:
    python analyze.py --json finite_states_machines/_01_LightSwitch_flattened.json
    python analyze.py --json finite_states_machines/_02_DimmableLightSwitch_flattened.json --episodes 2000
    python analyze.py --json finite_states_machines/_01_LightSwitch_flattened.json --all-fsms
"""

import argparse
import glob
import os
from collections import deque

import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para salvar gráficos
import matplotlib.pyplot as plt
import numpy as np

from fsm_loader import load_fsm_from_json
from q_learning import QLearningAgent
from random_agent import RandomAgent


# ===========================================================================
#  Utilidades
# ===========================================================================

def bfs_shortest_paths(fsm):
    """
    Calcula o caminho mais curto (BFS) de cada estado até qualquer
    estado-objetivo da FSM.

    Returns:
        Dicionário {estado: comprimento_do_caminho} ou None se inalcançável.
    """
    if not fsm.goal_states:
        return {}

    shortest = {}
    for goal in fsm.goal_states:
        # BFS reverso a partir do objetivo
        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            state = queue.popleft()
            for (s, a), t in fsm.transitions.items():
                if t == state and s not in dist:
                    dist[s] = dist[state] + 1
                    queue.append(s)
        for s, d in dist.items():
            if s not in shortest or d < shortest[s]:
                shortest[s] = d

    return shortest


def smooth(data, window=50):
    """Aplica média móvel para suavizar curvas."""
    if len(data) < window:
        window = max(1, len(data) // 5)
    return np.convolve(data, np.ones(window) / window, mode="valid")


# ===========================================================================
#  Gráfico 1: Curva de Aprendizado (Recompensa × Episódio)
# ===========================================================================

def plot_learning_curve(q_agent, r_agent, fsm_name, output_dir):
    """Gera gráfico de recompensa por episódio: Q-Learning vs Random."""
    fig, ax = plt.subplots(figsize=(10, 6))

    q_smooth = smooth(q_agent.rewards_per_episode)
    r_smooth = smooth(r_agent.rewards_per_episode)

    ax.plot(q_smooth, label="Q-Learning", color="#2196F3", linewidth=2)
    ax.plot(r_smooth, label="Random Agent", color="#F44336", linewidth=2, linestyle="--")

    ax.set_xlabel("Episódio", fontsize=12)
    ax.set_ylabel("Recompensa Acumulada (média móvel)", fontsize=12)
    ax.set_title(f"Curva de Aprendizado — {fsm_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f"learning_curve_{fsm_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [OK] {path}")


# ===========================================================================
#  Gráfico 2: Convergência de Cobertura
# ===========================================================================

def plot_coverage_convergence(q_agent, r_agent, fsm, fsm_name, output_dir):
    """Gera gráfico de cobertura de estados e transições × episódios."""
    total_states = len(fsm.states)
    total_transitions = len(fsm.transitions)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # -- State Coverage --
    q_sc = [s / total_states * 100 for s in q_agent.states_visited_per_episode]
    r_sc = [s / total_states * 100 for s in r_agent.states_visited_per_episode]

    ax1.plot(q_sc, label="Q-Learning", color="#2196F3", linewidth=2)
    ax1.plot(r_sc, label="Random Agent", color="#F44336", linewidth=2, linestyle="--")
    ax1.axhline(y=100, color="#4CAF50", linestyle=":", alpha=0.7, label="100%")
    ax1.set_xlabel("Episódio", fontsize=12)
    ax1.set_ylabel("State Coverage (%)", fontsize=12)
    ax1.set_title("Cobertura de Estados", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 110)

    # -- Transition Coverage --
    if total_transitions > 0:
        q_tc = [t / total_transitions * 100 for t in q_agent.transitions_visited_per_episode]
        r_tc = [t / total_transitions * 100 for t in r_agent.transitions_visited_per_episode]
    else:
        q_tc = [0] * len(q_agent.transitions_visited_per_episode)
        r_tc = [0] * len(r_agent.transitions_visited_per_episode)

    ax2.plot(q_tc, label="Q-Learning", color="#2196F3", linewidth=2)
    ax2.plot(r_tc, label="Random Agent", color="#F44336", linewidth=2, linestyle="--")
    ax2.axhline(y=100, color="#4CAF50", linestyle=":", alpha=0.7, label="100%")
    ax2.set_xlabel("Episódio", fontsize=12)
    ax2.set_ylabel("Transition Coverage (%)", fontsize=12)
    ax2.set_title("Cobertura de Transições", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 110)

    fig.suptitle(f"Convergência da Cobertura — {fsm_name}", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, f"coverage_convergence_{fsm_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


# ===========================================================================
#  Gráfico 3: Análise de Sensibilidade de Hiperparâmetros
# ===========================================================================

def plot_sensitivity_analysis(fsm, fsm_name, episodes, max_steps, output_dir):
    """
    Avalia o impacto de alpha, gamma e epsilon na cobertura de estados.
    Varia um parâmetro por vez, mantendo os outros fixos.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    configs = {
        "alpha": {"values": [0.01, 0.05, 0.1, 0.2, 0.5], "default": {"gamma": 0.9, "epsilon": 1.0}},
        "gamma": {"values": [0.5, 0.7, 0.9, 0.95, 0.99], "default": {"alpha": 0.1, "epsilon": 1.0}},
        "epsilon": {"values": [0.1, 0.3, 0.5, 0.8, 1.0], "default": {"alpha": 0.1, "gamma": 0.9}},
    }

    colors = ["#1565C0", "#42A5F5", "#66BB6A", "#FFA726", "#EF5350"]

    for idx, (param, config) in enumerate(configs.items()):
        ax = axes[idx]
        total_states = len(fsm.states)

        for i, val in enumerate(config["values"]):
            kwargs = dict(config["default"])
            kwargs[param] = val
            agent = QLearningAgent(
                alpha=kwargs.get("alpha", 0.1),
                gamma=kwargs.get("gamma", 0.9),
                epsilon=kwargs.get("epsilon", 1.0),
                epsilon_decay=0.995,
                epsilon_min=0.01,
            )
            agent.train(fsm, episodes=episodes, max_steps_per_episode=max_steps, verbose=False)

            sc = [s / total_states * 100 for s in agent.states_visited_per_episode]
            ax.plot(sc, label=f"{param}={val}", color=colors[i], linewidth=1.5)

        ax.axhline(y=100, color="#4CAF50", linestyle=":", alpha=0.5)
        ax.set_xlabel("Episódio", fontsize=11)
        ax.set_ylabel("State Coverage (%)", fontsize=11)
        ax.set_title(f"Sensibilidade a {param}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 110)

    fig.suptitle(f"Análise de Sensibilidade — {fsm_name}", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, f"sensitivity_{fsm_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


# ===========================================================================
#  Gráfico 4: Eficiência de Caminho (Q-Learning vs BFS)
# ===========================================================================

def plot_path_efficiency(fsm, q_agent, fsm_name, output_dir):
    """
    Compara o comprimento dos caminhos Q-Learning com o BFS ótimo.
    """
    bfs_paths = bfs_shortest_paths(fsm)
    if not bfs_paths:
        print(f"  [!] Sem estados-objetivo — pulando eficiência de caminho para {fsm_name}")
        return

    non_terminal = [s for s in fsm.states if not fsm.is_terminal(s) and s in bfs_paths]
    if not non_terminal:
        return

    states_labels = []
    q_lengths = []
    bfs_lengths = []

    for state in non_terminal:
        path = q_agent.get_optimal_path(fsm, state)
        q_len = len(path) - 1  # passos
        bfs_len = bfs_paths.get(state, 0)

        states_labels.append(state[:20])  # truncar labels longos
        q_lengths.append(q_len)
        bfs_lengths.append(bfs_len)

    fig, ax = plt.subplots(figsize=(max(8, len(non_terminal) * 0.6), 6))

    x = np.arange(len(states_labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, bfs_lengths, width, label="BFS (ótimo)", color="#4CAF50", alpha=0.8)
    bars2 = ax.bar(x + width / 2, q_lengths, width, label="Q-Learning", color="#2196F3", alpha=0.8)

    ax.set_ylabel("Passos até o objetivo", fontsize=12)
    ax.set_title(f"Eficiência de Caminho — {fsm_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(states_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = os.path.join(output_dir, f"path_efficiency_{fsm_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


# ===========================================================================
#  Tabela de Resumo Comparativo
# ===========================================================================

def print_summary_table(q_agent, r_agent, fsm, fsm_name):
    """Imprime tabela de resumo comparativo para a dissertação."""
    total_states = len(fsm.states)
    total_transitions = len(fsm.transitions)

    q_sc = q_agent.states_visited_per_episode[-1] / total_states * 100
    r_sc = r_agent.states_visited_per_episode[-1] / total_states * 100

    q_tc = q_agent.transitions_visited_per_episode[-1] / total_transitions * 100 if total_transitions else 0
    r_tc = r_agent.transitions_visited_per_episode[-1] / total_transitions * 100 if total_transitions else 0

    q_conv = q_agent.get_convergence_episode(total_states, threshold=0.9)
    r_conv = r_agent.get_convergence_episode(total_states, threshold=0.9)

    q_avg_reward = sum(q_agent.rewards_per_episode[-100:]) / min(100, len(q_agent.rewards_per_episode))
    r_avg_reward = sum(r_agent.rewards_per_episode[-100:]) / min(100, len(r_agent.rewards_per_episode))

    print(f"\n{'=' * 65}")
    print(f"  RESUMO COMPARATIVO — {fsm_name}")
    print(f"{'=' * 65}")
    print(f"  {'Métrica':<35} {'Q-Learning':>12} {'Random':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    print(f"  {'State Coverage (%)' :<35} {q_sc:>11.1f}% {r_sc:>11.1f}%")
    print(f"  {'Transition Coverage (%)':<35} {q_tc:>11.1f}% {r_tc:>11.1f}%")
    print(f"  {'Episódio convergência (90%)':<35} {str(q_conv or 'N/A'):>12} {str(r_conv or 'N/A'):>12}")
    print(f"  {'Recompensa média (últ. 100)':<35} {q_avg_reward:>12.1f} {r_avg_reward:>12.1f}")
    print(f"{'=' * 65}\n")


# ===========================================================================
#  Análise Principal
# ===========================================================================

def analyze_fsm(json_path, episodes, max_steps, output_dir):
    """Executa toda a análise para uma FSM."""
    fsm_name = os.path.splitext(os.path.basename(json_path))[0]
    fsm = load_fsm_from_json(json_path)

    print(f"\n{'=' * 65}")
    print(f"  ANALISANDO: {fsm_name}")
    print(f"  Estados: {len(fsm.states)} | Transições: {len(fsm.transitions)}")
    print(f"  Episódios: {episodes} | Max passos: {max_steps}")
    print(f"{'=' * 65}")

    # 1. Treinar Q-Learning
    print("\n[1/5] Treinando Q-Learning...")
    q_agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)
    q_agent.train(fsm, episodes=episodes, max_steps_per_episode=max_steps, verbose=False)

    # 2. Treinar Random Agent
    print("[2/5] Executando Random Agent...")
    r_agent = RandomAgent()
    r_agent.train(fsm, episodes=episodes, max_steps_per_episode=max_steps, verbose=False)

    # 3. Gerar gráficos
    print("[3/5] Gerando curva de aprendizado...")
    plot_learning_curve(q_agent, r_agent, fsm_name, output_dir)

    print("[4/5] Gerando convergência de cobertura...")
    plot_coverage_convergence(q_agent, r_agent, fsm, fsm_name, output_dir)

    print("[5/5] Gerando análise de sensibilidade...")
    plot_sensitivity_analysis(fsm, fsm_name, episodes, max_steps, output_dir)

    # 4. Eficiência de caminho (só se houver objetivos)
    if fsm.goal_states:
        plot_path_efficiency(fsm, q_agent, fsm_name, output_dir)

    # 5. Tabela resumo
    print_summary_table(q_agent, r_agent, fsm, fsm_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Análise Quantitativa — Q-Learning vs Random Agent para FSM"
    )
    parser.add_argument(
        "--json", type=str,
        help="Caminho para o arquivo JSON da FSM.",
    )
    parser.add_argument(
        "--all-fsms", action="store_true",
        help="Analisar todas as FSMs em finite_states_machines/.",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="Episódios de treinamento (padrão: 1000).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Máximo de passos por episódio (padrão: 50).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Diretório para salvar os gráficos (padrão: results/).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Criar diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  ANÁLISE QUANTITATIVA — Q-LEARNING PARA FSM")
    print("  Gráficos para Dissertação de Mestrado")
    print("=" * 65)

    if args.all_fsms:
        fsm_dir = os.path.join(os.path.dirname(__file__), "finite_states_machines")
        json_files = sorted(glob.glob(os.path.join(fsm_dir, "*.json")))
        if not json_files:
            print("Nenhum arquivo JSON encontrado em finite_states_machines/")
            return
        for json_path in json_files:
            analyze_fsm(json_path, args.episodes, args.max_steps, args.output_dir)
    elif args.json:
        analyze_fsm(args.json, args.episodes, args.max_steps, args.output_dir)
    else:
        print("Erro: use --json <arquivo> ou --all-fsms")
        return

    print(f"\n[OK] Todos os graficos salvos em: {os.path.abspath(args.output_dir)}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
