import gymnasium as gym
import numpy as np
import imageio
import os

BINS = 10
low  = np.array([-2.4, -4, -0.2095, -4])
high = np.array([ 2.4,  4,  0.2095,  4])

def discretize(obs):
    ratios  = (obs - low) / (high - low)
    indices = np.clip((ratios * BINS).astype(int), 0, BINS - 1)
    return tuple(indices)

q_table = np.load("results/q_table.npy")

env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, _ = env.reset(seed=42)
state = discretize(obs)
frames = []

for _ in range(500):
    frames.append(env.render())
    action = np.argmax(q_table[state])
    obs, _, terminated, truncated, _ = env.step(action)
    state = discretize(obs)
    if terminated or truncated:
        break

env.close()

os.makedirs("results", exist_ok=True)
imageio.mimsave("results/agent.gif", frames, fps=30)
print(f"Saved results/agent.gif ({len(frames)} frames)")
