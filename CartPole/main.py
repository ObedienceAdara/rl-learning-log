import gymnasium as gym
import numpy as np
import csv
import os

# --- Hyperparameters ---
EPISODES       = 500
LEARNING_RATE  = 0.1
DISCOUNT       = 0.95
EPSILON        = 1.0
EPSILON_DECAY  = 0.995
MIN_EPSILON    = 0.01
BINS           = 10
THRESHOLD      = 475  # reward considered "solved"
EVAL_EPISODES  = 10
SEED           = 42

# --- Discretize continuous observation into bins ---
# obs: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
low  = np.array([-2.4, -4, -0.2095, -4])
high = np.array([ 2.4,  4,  0.2095,  4])

def discretize(obs):
    ratios  = (obs - low) / (high - low)
    indices = np.clip((ratios * BINS).astype(int), 0, BINS - 1)
    return tuple(indices)

# --- Training ---
def train():
    env = gym.make("CartPole-v1", render_mode="human")
    q_table = np.zeros([BINS] * 4 + [2])
    epsilon = EPSILON
    episode_rewards = []

    for episode in range(EPISODES):
        obs, _ = env.reset(seed=SEED + episode)
        state = discretize(obs)
        total_reward = 0

        for _ in range(500):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs)

            # Q-update (zero out future value on terminal state)
            if terminated:
                target = reward
            else:
                target = reward + DISCOUNT * np.max(q_table[next_state])

            q_table[state][action] += LEARNING_RATE * (target - q_table[state][action])

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1:4d} | Reward: {total_reward:6.1f} | Epsilon: {epsilon:.3f}")

    env.close()
    return q_table, episode_rewards

# --- Evaluation (no exploration) ---
def evaluate(q_table):
    env = gym.make("CartPole-v1", render_mode="human")
    eval_rewards = []

    for i in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=i)
        state = discretize(obs)
        total_reward = 0

        for _ in range(500):
            action = np.argmax(q_table[state])
            obs, reward, terminated, truncated, _ = env.step(action)
            state = discretize(obs)
            total_reward += reward
            if terminated or truncated:
                break

        eval_rewards.append(total_reward)

    env.close()
    return eval_rewards

# --- Save rewards to CSV ---
def save_csv(episode_rewards):
    os.makedirs("results", exist_ok=True)
    with open("results/training_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(episode_rewards, start=1):
            writer.writerow([i, r])

# --- Compute episodes to threshold (first 50-ep rolling mean >= THRESHOLD) ---
def episodes_to_threshold(rewards):
    for i in range(50, len(rewards) + 1):
        if np.mean(rewards[i - 50:i]) >= THRESHOLD:
            return i
    return None

# --- Main ---
if __name__ == "__main__":
    print("Training...\n")
    q_table, episode_rewards = train()
    save_csv(episode_rewards)
    np.save("results/q_table.npy", q_table)

    print("\nEvaluating...\n")
    eval_rewards = evaluate(q_table)

    mean_eval  = np.mean(eval_rewards)
    std_eval   = np.std(eval_rewards)
    best_ep    = max(episode_rewards)
    threshold  = episodes_to_threshold(episode_rewards)
    success    = sum(r >= THRESHOLD for r in eval_rewards) / EVAL_EPISODES

    print("\n┌─────────────────────────────────────────┐")
    print(f"│  Mean Eval Reward      {mean_eval:6.1f} ± {std_eval:.1f}        │")
    print(f"│  Best Training Episode {best_ep:6.1f}            │")
    print(f"│  Episodes to Threshold {str(threshold) + ' ep':>10}           │")
    print(f"│  Success Rate          {success * 100:5.1f}%            │")
    print(f"│  Training Runs         1                  │")
    print("└─────────────────────────────────────────┘")
