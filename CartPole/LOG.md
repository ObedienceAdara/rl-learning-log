# CartPole Log

## v1
Started with the basic random agent from the Gymnasium docs. The environment is `CartPole-v1`, where the goal is to balance a pole on a cart. Added `render_mode="human"` to visualize it in a window. The agent picks random actions (push left or right) with no learning involved, so the pole falls quickly each episode.

## v2
Replaced the random agent with a Q-learning implementation. Since CartPole's observation space is continuous, we discretize it into 10 bins per dimension to build a finite Q-table. The agent uses epsilon-greedy exploration, starting fully random and gradually exploiting learned Q-values as epsilon decays. After each step, the Q-table is updated using the Bellman equation. The agent improves over 500 episodes, with total reward increasing as it learns to balance the pole longer.

## v3
Evolved the implementation into a full experiment pipeline. `main.py` now trains the agent with a corrected Q-update (zeroing future value on terminal states), evaluates the learned policy over 10 episodes with no exploration, and prints a summary of all key metrics. Results are saved to `results/` — training rewards as a CSV, the Q-table as a `.npy` file. Added `plot.py` to generate a reward curve with a 25-episode rolling mean, and `record.py` to load the saved Q-table and record an agent demo as a GIF. These artifacts feed directly into the portfolio `/lab` page.
