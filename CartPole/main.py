import gymnasium as gym
import numpy as np

# --- Hyperparameters ---
EPISODES = 500
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BINS = 10  

# --- Discretize the continuous observation space ---
# CartPole obs: [cart_pos, cart_vel, pole_angle, pole_vel]
low  = [-2.4, -4, -0.2095, -4]
high = [ 2.4,  4,  0.2095,  4]

def discretize(observation):
    ratios = (observation - low) / (np.array(high) - np.array(low))
    indices = np.clip((ratios * BINS).astype(int), 0, BINS - 1)
    return tuple(indices)

# Q-table: 10 bins per dimension, 2 actions
q_table = np.zeros([BINS] * 4 + [2])

env = gym.make("CartPole-v1", render_mode="human")

epsilon = EPSILON

for episode in range(EPISODES):
    obs, _ = env.reset()
    state = discretize(obs)
    total_reward = 0

    for _ in range(500):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(q_table[state])  # exploit

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(next_obs)

        # Q-learning update
        best_next = np.max(q_table[next_state])
        q_table[state][action] += LEARNING_RATE * (
            reward + DISCOUNT * best_next - q_table[state][action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")

env.close()
