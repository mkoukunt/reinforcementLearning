import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Outputs valid probability distribution
        )

    def forward(self, state):
        return self.network(state)


def train_policy_gradient():
    # 1. Environment & Hyperparameters initialization
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    gamma = 0.99
    learning_rate = 0.01
    num_episodes = 500

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)  #

    for episode in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        truncated = False

        # 2. Collect a full episode trajectory
        while not (done or truncated):
            state_t = torch.FloatTensor(state)
            action_probs = policy(state_t)

            # Create a categorical distribution to sample actions
            dist = Categorical(action_probs)
            action = dist.sample()

            # Save the log probability for the gradient calculation
            log_probs.append(dist.log_prob(action))

            # Step the environment
            state, reward, done, truncated, _ = env.step(action.item())
            rewards.append(reward)

        # 3. Calculate discounted returns (Rewards-to-go)
        discounted_returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_returns.insert(0, G)

        # Normalize returns for training stability
        returns_t = torch.FloatTensor(discounted_returns)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # 4. Compute policy loss
        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns_t):
            # Negative sign turns gradient descent into gradient ascent
            policy_loss.append(-log_prob * G_t)

        # Stack losses and calculate gradients
        optimizer.zero_grad()  #
        loss = torch.stack(policy_loss).sum()
        loss.backward()  #
        optimizer.step()  #

        # Diagnostic printout
        total_reward = sum(rewards)
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1:3d} | Total Reward: {total_reward:.1f}")


    env.close()
    with torch.no_grad():
        action_probs = policy(state_t)
        print(f"Action Probabilities: {action_probs.numpy()}")


if __name__ == "__main__":
    train_policy_gradient()
