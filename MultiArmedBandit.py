import numpy as np

class MultiArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.q_values = np.zeros(n_arms)  # Estimated value of each arm
        self.counts = np.zeros(n_arms)    # Number of times each arm was pulled

    def select_action(self, epsilon):
        # Exploration: pick a random arm
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_arms)
        # Exploitation: pick the arm with the highest estimated value
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        self.counts[action] += 1
        # Incremental update rule: Q(n+1) = Q(n) + (Reward - Q(n)) / n
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action]

# Setup
n_arms = 5
true_probabilities = [0.1, 0.4, 0.2, 0.8, 0.5]  # Hidden true win rates
bandit_agent = MultiArmedBandit(n_arms)
epsilon = 0.1  # 10% explore, 90% exploit

# Simulate pulling a lever
def pull_lever(arm_index):
    # Returns a reward (1) or no reward (0)
    return 1 if np.random.rand() < true_probabilities[arm_index] else 0

# Training loop example
n_trials = 5000
for step in range(n_trials):
    action = bandit_agent.select_action(epsilon)
    reward = pull_lever(action)
    bandit_agent.update(action, reward)

print("Estimated values:", bandit_agent.q_values)
print("Best arm to pull:", np.argmax(bandit_agent.q_values))
