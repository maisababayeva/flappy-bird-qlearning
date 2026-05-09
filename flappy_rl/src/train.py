"""
Training script for the Flappy Bird Q-Learning agent.

Runs three experimental configurations:
    1. Baseline      – standard settings
    2. Fast pipes    – higher pipe speed

    (2. Narrow gap    – harder environment (smaller pipe gap)) - old

For each config, the agent is trained and results are saved.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
import time

from flappy_env import FlappyBirdEnv, PIPE_GAP, PIPE_VX
from agent import QLearningAgent


# ── Training loop ─────────────────────────────────────────────────────────

def train(
    env: FlappyBirdEnv,
    agent: QLearningAgent,
    n_episodes: int = 5000,
    log_interval: int = 500,
    label: str = "run",
) -> dict:
    """
    Train the agent on the given environment.
    Returns a dict of per-episode metrics.
    """
    scores        = []
    rewards_total = []
    epsilons      = []
    frames_alive  = []

    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        state      = env.reset()
        done       = False
        ep_reward  = 0.0
        ep_frames  = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state      = next_state
            ep_reward += reward
            ep_frames += 1

        agent.decay_epsilon()

        scores.append(info["score"])
        rewards_total.append(ep_reward)
        epsilons.append(agent.epsilon)
        frames_alive.append(ep_frames)

        if ep % log_interval == 0:
            recent = scores[-log_interval:]
            elapsed = time.time() - t0
            print(
                f"[{label}] Ep {ep:>5}/{n_episodes}  "
                f"avg_score={np.mean(recent):.2f}  "
                f"max_score={max(recent)}  "
                f"epsilon={agent.epsilon:.3f}  "
                f"Q-states={agent.q_table_size()}  "
                f"time={elapsed:.1f}s"
            )

    return {
        "scores":        scores,
        "rewards_total": rewards_total,
        "epsilons":      epsilons,
        "frames_alive":  frames_alive,
    }


# ── Evaluation (greedy policy) ─────────────────────────────────────────────

def evaluate(env: FlappyBirdEnv, agent: QLearningAgent, n_episodes: int = 200) -> dict:
    """Run greedy policy and collect stats."""
    scores = []
    frames = []
    for _ in range(n_episodes):
        state = env.reset()
        done  = False
        fr    = 0
        while not done:
            action = agent.choose_action_greedy(state)
            state, _, done, info = env.step(action)
            fr += 1
        scores.append(info["score"])
        frames.append(fr)

    return {
        "mean_score":   float(np.mean(scores)),
        "max_score":    int(max(scores)),
        "median_score": float(np.median(scores)),
        "std_score":    float(np.std(scores)),
        "mean_frames":  float(np.mean(frames)),
        "n_episodes":   n_episodes,
    }


# ── Experiments ───────────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        "label":      "Baseline",
        "pipe_gap":   PIPE_GAP,           # 150
        "pipe_speed": PIPE_VX,            # -4.0
        "n_episodes": 8000,
        "alpha":      0.1,
        "gamma":      0.99,
        "epsilon_decay": 0.0005,
    },
    {
        "label":      "Fast_Pipes",
        "pipe_gap":   PIPE_GAP,
        "pipe_speed": -6.0,               # faster pipes
        "n_episodes": 8000,
        "alpha":      0.1,
        "gamma":      0.99,
        "epsilon_decay": 0.0005,
    },
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")


def run_all():
    all_results = {}

    for cfg in EXPERIMENTS:
        label = cfg["label"]
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {label}")
        print(f"  pipe_gap={cfg['pipe_gap']}  pipe_speed={cfg['pipe_speed']}")
        print(f"{'='*60}")

        env = FlappyBirdEnv(pipe_gap=cfg["pipe_gap"], pipe_speed=cfg["pipe_speed"])
        agent = QLearningAgent(
            alpha=cfg["alpha"],
            gamma=cfg["gamma"],
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=cfg["epsilon_decay"],
        )

        metrics = train(env, agent, n_episodes=cfg["n_episodes"], label=label)

        # Evaluate trained agent
        eval_env = FlappyBirdEnv(pipe_gap=cfg["pipe_gap"], pipe_speed=cfg["pipe_speed"])
        eval_stats = evaluate(eval_env, agent, n_episodes=200)
        print(f"\n[{label}] Eval → {eval_stats}")

        # Save model
        agent.save(os.path.join(MODELS_DIR, f"agent_{label}.pkl"))

        # Save metrics
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_path = os.path.join(RESULTS_DIR, f"metrics_{label}.json")
        with open(results_path, "w") as f:
            json.dump({"train": metrics, "eval": eval_stats, "config": cfg}, f)

        all_results[label] = {"train": metrics, "eval": eval_stats, "config": cfg}

    # Save combined
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f)

    print("\n✓ All experiments complete. Results saved.")
    return all_results


if __name__ == "__main__":
    run_all()
