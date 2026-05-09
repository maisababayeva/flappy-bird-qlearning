"""
Generate all analysis plots for the Flappy Bird RL project.
Saves figures to results/.
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

EXPERIMENTS = {
    "Baseline": {
        "color": "#2196F3",
        "dark_color": "#1565C0",
        "label": "Baseline",
        "title": "Baseline (gap=150, speed=4)",
        "gap": 150,
        "speed": -4.0,
    },
    "Fast_Pipes": {
        "color": "#4CAF50",
        "dark_color": "#1B5E20",
        "label": "Fast Pipes",
        "title": "Fast Pipes (gap=150, speed=6)",
        "gap": 150,
        "speed": -6.0,
    },
}


def smooth(data, window=100):
    if len(data) < window:
        return data

    return np.convolve(data, np.ones(window) / window, mode="valid")


def load_results():
    path = os.path.join(RESULTS_DIR, "all_results.json")

    with open(path) as f:
        return json.load(f)


def fig1_learning_curves(results):
    """Smoothed average score over training episodes."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, (key, cfg) in zip(axes, EXPERIMENTS.items()):
        scores = results[key]["train"]["scores"]

        sm = smooth(scores, window=200)
        ep = np.arange(len(sm)) + 200

        ax.plot(ep, sm, color=cfg["color"], linewidth=2)
        ax.fill_between(ep, sm, alpha=0.15, color=cfg["color"])

        ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Score (200-episode window)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(scores))
        ax.set_ylim(bottom=0)

    plt.suptitle(
        "Learning Curves - Q-Learning Agent on Flappy Bird",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "fig1_learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {path}")


def fig2_epsilon_decay(results):
    """Epsilon decay over episodes."""

    fig, ax = plt.subplots(figsize=(9, 4))

    for key, cfg in EXPERIMENTS.items():
        eps = results[key]["train"]["epsilons"]

        ax.plot(
            eps,
            color=cfg["color"],
            linewidth=2,
            label=cfg["label"],
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon (Exploration Rate)")
    ax.set_title("Epsilon Decay Over Training", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "fig2_epsilon_decay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {path}")


def fig3_score_distribution(results):
    """Box plot of evaluation scores per experiment."""

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

    from flappy_env import FlappyBirdEnv
    from agent import QLearningAgent

    all_scores = []
    labels_short = []

    for key, cfg in EXPERIMENTS.items():
        model_path = os.path.join(MODELS_DIR, f"agent_{key}.pkl")

        agent = QLearningAgent.load(model_path)
        env = FlappyBirdEnv(pipe_gap=cfg["gap"], pipe_speed=cfg["speed"])

        scores = []

        for _ in range(300):
            state = env.reset()
            done = False

            while not done:
                action = agent.choose_action_greedy(state)
                state, _, done, info = env.step(action)

            scores.append(info["score"])

        all_scores.append(scores)
        labels_short.append(cfg["label"])

    fig, ax = plt.subplots(figsize=(7, 5))

    bp = ax.boxplot(
        all_scores,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )

    for patch, cfg in zip(bp["boxes"], EXPERIMENTS.values()):
        patch.set_facecolor(cfg["color"])
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels_short)
    ax.set_ylabel("Score (pipes passed)")
    ax.set_title(
        "Evaluation Score Distribution (300 episodes per config)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "fig3_score_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {path}")

    return all_scores


def fig4_comparison_bar(results):
    """Bar chart comparing evaluation metrics."""

    keys = list(EXPERIMENTS.keys())
    labels = [EXPERIMENTS[k]["label"] for k in keys]

    means = [results[k]["eval"]["mean_score"] for k in keys]
    maxs = [results[k]["eval"]["max_score"] for k in keys]
    stds = [results[k]["eval"]["std_score"] for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    colors_mean = [EXPERIMENTS[k]["color"] for k in keys]
    colors_max = [EXPERIMENTS[k]["dark_color"] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(
        x - width / 2,
        means,
        width,
        label="Mean Score",
        color=colors_mean,
        alpha=0.85,
        yerr=stds,
        capsize=5,
        error_kw={"elinewidth": 1.5},
    )

    bars2 = ax.bar(
        x + width / 2,
        maxs,
        width,
        label="Max Score",
        color=colors_max,
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score (pipes passed)")
    ax.set_title("Evaluation Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "fig4_comparison_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {path}")


def fig5_survival_frames(results):
    """Average frames alive over training."""

    fig, ax = plt.subplots(figsize=(10, 4))

    for key, cfg in EXPERIMENTS.items():
        frames = results[key]["train"]["frames_alive"]

        sm = smooth(frames, window=200)
        ep = np.arange(len(sm)) + 200

        ax.plot(
            ep,
            sm,
            color=cfg["color"],
            linewidth=2,
            label=cfg["label"],
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Frames Survived (200-episode window)")
    ax.set_title("Survival Time During Training", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "fig5_survival_frames.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {path}")


def print_summary_table(results):
    """Print a simple summary table in the terminal."""

    print("\n" + "=" * 70)
    print(f"{'Config':<25} {'Mean':>8} {'Max':>8} {'Median':>8} {'Std':>8}")
    print("-" * 70)

    for key, cfg in EXPERIMENTS.items():
        e = results[key]["eval"]

        print(
            f"{cfg['title']:<25} "
            f"{e['mean_score']:>8.2f} "
            f"{e['max_score']:>8} "
            f"{e['median_score']:>8.1f} "
            f"{e['std_score']:>8.2f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    results = load_results()

    fig1_learning_curves(results)
    fig2_epsilon_decay(results)
    fig3_score_distribution(results)
    fig4_comparison_bar(results)
    fig5_survival_frames(results)

    print_summary_table(results)

    print("\n✓ All figures saved.")