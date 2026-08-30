import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/training_rewards.csv")
df["rolling"] = df["reward"].rolling(25).mean()

plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["reward"], alpha=0.25, label="Episode reward")
plt.plot(df["episode"], df["rolling"], label="25-ep rolling mean")
plt.axhline(475, color="red", linestyle="--", linewidth=0.8, label="Solved threshold (475)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("CartPole-v1 — Tabular Q-Learning")
plt.legend()
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/reward_curve.png", dpi=200)
plt.close()

print("Saved results/reward_curve.png")
