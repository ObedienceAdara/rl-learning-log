from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

from src.gridworld import START, NUM_ACTIONS, NUM_STATES, step

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 42
EPISODES = 500
MAX_STEPS_PER_EPISODE = 200
ALPHA = 0.10
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01


def choose_action(q_table: np.ndarray, state: int, epsilon: float) -> int:
    """Choose an action with an epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.randrange(NUM_ACTIONS)
    return int(np.argmax(q_table[state]))


def train() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    q_table = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)
    epsilon = EPSILON_START
    episode_records: list[dict] = []

    for episode in range(1, EPISODES + 1):
        state = START
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = choose_action(q_table, state, epsilon)
            transition = step(state, action)

            old_value = q_table[state, action]
            if transition.done:
                target = transition.reward
            else:
                target = transition.reward + GAMMA * np.max(q_table[transition.next_state])

            q_table[state, action] = old_value + ALPHA * (target - old_value)

            state = transition.next_state
            total_reward += transition.reward
            steps += 1
            done = transition.done

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        episode_records.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": steps,
                "epsilon": epsilon,
                "success": int(done),
            }
        )

    np.save(DATA_DIR / "q_table_500_episodes.npy", q_table)

    metadata = {
        "episodes": EPISODES,
        "seed": SEED,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon_start": EPSILON_START,
        "epsilon_decay": EPSILON_DECAY,
        "epsilon_min": EPSILON_MIN,
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "num_states": NUM_STATES,
        "num_actions": NUM_ACTIONS,
        "start_state": START,
    }
    (DATA_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    with (DATA_DIR / "episode_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=episode_records[0].keys())
        writer.writeheader()
        writer.writerows(episode_records)

    policy = {str(state): int(np.argmax(q_table[state])) for state in range(NUM_STATES)}
    (DATA_DIR / "greedy_policy.json").write_text(json.dumps(policy, indent=2))

    success_rate = sum(record["success"] for record in episode_records) / EPISODES
    print(f"Training complete: {EPISODES} episodes")
    print(f"Final epsilon: {epsilon:.4f}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Saved: {DATA_DIR / 'q_table_500_episodes.npy'}")
    print(f"Saved: {DATA_DIR / 'greedy_policy.json'}")


if __name__ == "__main__":
    train()
