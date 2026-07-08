import random
import numpy as np
import gymnasium as gym
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import GridWorldEnv
# --- Hyperparameters ---
ENV_NAME = 'gymnasium_env/GridWorld-v0'#"CartPole-v1"
GAMMA = 0.99  # Discount factor for future rewards
LEARNING_RATE = 1e-3  # Optimizer learning rate
MEMORY_SIZE = 10_000  # Maximum size of replay buffer
BATCH_SIZE = 64  # Size of batch sampled from replay buffer
EPSILON_START = 1.0  # Initial exploration probability
EPSILON_END = 0.05  # Minimum exploration probability
EPSILON_DECAY = 0.995  # Decay rate per episode
TARGET_UPDATE = 10  # Frequency (in episodes) to sync target network


# --- 1. Q-Network Architecture ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.network(x)
def obs_to_tensor1(obs):
    # Flatten the Dict observation ({"agent": [x, y], "target": [x, y]})
    # into a single 1D float tensor for the policy network.
    return torch.FloatTensor(
        np.concatenate([obs["agent"], obs["target"]]).astype(np.float32)
    )
def obs_to_tensor(obs):
    # Iterate over the tuple of Dict observations
    # ({"agent": [x, y], "target": [x, y]}), flattening each into a
    # 4-element [agent_x, agent_y, target_x, target_y] row, and stack
    # them into a (64, 4) float tensor for the policy network.
    rows = [
        np.concatenate([o["agent"], o["target"]]).astype(np.float32)
        for o in obs
    ]
    return torch.FloatTensor(np.stack(rows))

# --- 2. Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))



    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (obs_to_tensor(state),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                obs_to_tensor(next_state),
                torch.FloatTensor(done))


    def __len__(self):
        return len(self.buffer)


# --- 3. DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START

        # Online and target networks
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network does not need gradients

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)

    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_t = obs_to_tensor1(state).unsqueeze(0)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample mini-batch from experience replay
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Compute predicted Q values for taken actions
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values using the Target Network (Bellman Equation)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (GAMMA * max_next_q * (1 - dones))

        # Optimize using Mean Squared Error loss
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# --- 4. Main Training Loop ---
if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    #state_dim = env.observation_space.shape[0]
    state_dim = sum(space.shape[0] for space in env.observation_space.spaces.values())
    action_dim = env.action_space.n


    agent = DQNAgent(state_dim, action_dim)
    num_episodes = 1000

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition in buffer
            agent.memory.push(state, action, reward, next_state, done)

            # Update policy network weights
            agent.update()

            state = next_state
            episode_reward += reward

        agent.decay_epsilon()

        # Synchronize target network periodically
        if episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:3d} | Total Reward: {episode_reward:5.1f} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    print("Training Complete!")
    with torch.no_grad():
        state, _ = env.reset()
        state = obs_to_tensor1(state)
        a=agent.target_net(state).argmax().item()
        print(a)
        state, reward, done, truncated, _ = env.step(a)
        state = obs_to_tensor1(state)
        a = agent.target_net(state).argmax().item()
        print(a,state)
        state, reward, done, truncated, _ = env.step(a)
        state = obs_to_tensor1(state)
        a = agent.target_net(state).argmax().item()
        print(a,state)
        state, reward, done, truncated, _ = env.step(a)
        state = obs_to_tensor1(state)
        a = agent.target_net(state).argmax().item()
        print(a, state)
        state, reward, done, truncated, _ = env.step(a)
        state = obs_to_tensor1(state)
        a = agent.target_net(state).argmax().item()
        print(a, state)
        state, reward, done, truncated, _ = env.step(a)
        state = obs_to_tensor1(state)
        a = agent.target_net(state).argmax().item()
        print(a,state)
        state, reward, done, truncated, _ = env.step(a)
        state = obs_to_tensor1(state)
        a = agent.target_net(state).argmax().item()
        print(a,state)
        state, reward, done, truncated, _ = env.step(a)
        state = obs_to_tensor1(state)
        a = agent.target_net(state).argmax().item()
        print(a, state)
        state, reward, done, truncated, _ = env.step(a)
        state = obs_to_tensor1(state)
        a = agent.target_net(state).argmax().item()
        print(a, state)


