import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym


class NeuralWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NeuralWorldModel, self).__init__()
        # Combine state and action dimensions for input
        input_dim = state_dim + action_dim

        # Shared feature extractor
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Head to predict the state delta (s_{t+1} - s_t) for better stability
        self.state_delta_head = nn.Linear(128, state_dim)
        # Head to predict immediate reward
        self.reward_head = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        features = self.network(x)

        state_delta = self.state_delta_head(features)
        reward = self.reward_head(features)

        return state_delta, reward


def collect_random_data(env_name, num_steps=2000):
    env = gym.make(env_name)
    states, actions, next_states, rewards = [], [], [], []

    state, _ = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)

        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    env.close()
    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1))


def train_world_model(model, data, epochs=50, batch_size=64):
    states, actions, next_states, rewards = [torch.tensor(d) for d in data]
    # Target values for the model output
    target_deltas = next_states - states

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    dataset_size = states.size(0)

    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i + batch_size]

            # Forward pass
            pred_deltas, pred_rewards = model(states[indices], actions[indices])

            # Loss calculation
            loss_state = loss_fn(pred_deltas, target_deltas[indices])
            loss_reward = loss_fn(pred_rewards, rewards[indices])
            total_loss = loss_state + loss_reward

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


class ModelBasedEnv(gym.Env):
    def __init__(self, world_model, real_env_name):
        super(ModelBasedEnv, self).__init__()
        self.model = world_model
        self.model.eval()  # Set network to evaluation mode

        # Mirror the exact spaces of the original environment
        real_env = gym.make(real_env_name)
        self.observation_space = real_env.observation_space
        self.action_space = real_env.action_space
        real_env.close()

        self.current_state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # In a strict model environment, we start from a default state
        # or sample a valid starting state from your data.
        self.current_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        return self.current_state, {}

    def step(self, action):
        # Convert inputs to torch tensors
        s_t = torch.tensor(self.current_state, dtype=torch.float32)

        # Handle cases for discrete actions or single continuous actions
        if isinstance(self.action_space, gym.spaces.Discrete):
            a_t = torch.tensor([action], dtype=torch.float32)
        else:
            a_t = torch.tensor(action, dtype=torch.float32)

        with torch.no_grad():
            pred_delta, pred_reward = self.model(s_t, a_t)

        # Reconstruct next state: s_{t+1} = s_t + delta
        next_state = (s_t + pred_delta).numpy()
        reward = float(pred_reward.item())

        # Update internal tracking
        self.current_state = next_state

        # Define termination logic based on state conditions if known,
        # or train an explicit termination head alongside the reward head.
        terminated = False
        truncated = False

        return next_state, reward, terminated, truncated, {}

# Configuration
ENV_NAME = "Pendulum-v1"
state_size = 3   # Pendulum observation space shape
action_size = 1  # Pendulum action space shape

# Pipeline Execution
print("Collecting buffer data from real world...")
raw_data = collect_random_data(ENV_NAME, num_steps=3000)

print("Initializing and training Neural Network World Model...")
world_model = NeuralWorldModel(state_dim=state_size, action_dim=action_size)
train_world_model(world_model, raw_data, epochs=40)

print("Deploying Simulated Model Environment...")
simulated_env = ModelBasedEnv(world_model, ENV_NAME)

# Test run with random inputs in the imagined world
obs, _ = simulated_env.reset()
print("Initial Imagined State:", obs)
next_obs, reward, _, _, _ = simulated_env.step(simulated_env.action_space.sample())
print("Next Imagined State:   ", next_obs)
print("Predicted Reward:      ", reward)
