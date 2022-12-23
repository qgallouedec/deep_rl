from typing import Any, Dict, Tuple

import gym
import numpy as np
import torch
from torch import Tensor, nn, optim


class TorchWrapper(gym.Wrapper):
    """
    Torch wrapper. Actions and observations are Tensors instead of arrays.
    """

    def step(self, action: Tensor) -> Tuple[Tensor, float, bool, Dict[str, Any]]:
        action = action.cpu().numpy()
        observation, reward, done, info = self.env.step(action)
        return torch.tensor(observation), reward, done, info

    def reset(self) -> Tensor:
        observation = self.env.reset()
        return torch.tensor(observation)


class QNetwork(nn.Module):
    def __init__(self, env: gym.Env) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, observation: Tensor) -> Tensor:
        return self.network(observation)


env_id = "CartPole-v1"

total_timesteps = 100_000
learning_starts = 10_000

start_e = 1
end_e = 0.05
exploration_fraction = 0.5
slope = (end_e - start_e) / (exploration_fraction * total_timesteps)

max_priority = 1.0
beta_increment_per_sampling = 0.001
alpha = 0.6

train_frequency = 10
batch_size = 128
gamma = 0.99
learning_rate = 2.5e-4
target_network_frequency = 500

# Env setup
env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
env = TorchWrapper(env)

# Seeding
seed = 1
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
env.action_space.seed(seed)

# Network setup
q_network = QNetwork(env)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
target_network = QNetwork(env)
target_network.load_state_dict(q_network.state_dict())

# Storage setup
observations = torch.zeros((total_timesteps + 1, *env.observation_space.shape))
actions = torch.zeros((total_timesteps + 1, *env.action_space.shape), dtype=torch.long)
rewards = torch.zeros((total_timesteps + 1))
terminated = torch.zeros((total_timesteps + 1), dtype=torch.bool)
priorities = torch.zeros((total_timesteps + 1), dtype=torch.float32)

# Initialize beta
beta = 0.4

# Initiate the envrionment and store the inital observation
observation = env.reset()
global_step = 0
observations[global_step] = observation

# Loop
while global_step < total_timesteps:
    # Update exploration rate
    epsilon = max(slope * global_step + start_e, end_e)

    if np.random.random() < epsilon:
        action = torch.tensor(env.action_space.sample())
    else:
        q_values = q_network(observation)
        action = torch.argmax(q_values)

    # Store
    actions[global_step] = action

    # Compute and store priority
    priorities[global_step] = max(torch.max(priorities), 1.0)

    # Step
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()

    # Update count
    global_step += 1

    # Store
    observations[global_step] = observation
    rewards[global_step] = reward
    terminated[global_step] = done and not info.get("TimeLimit.truncated", False)

    if "episode" in info.keys():
        print(f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}")

    # Optimize the agent
    if global_step >= learning_starts:
        if global_step % train_frequency == 0:
            probabilities = priorities**alpha / torch.sum(priorities**alpha)
            weights = 1 / (total_timesteps * probabilities) ** beta
            weights = weights / torch.max(weights)

            batch_inds = torch.multinomial(priorities, batch_size, replacement=True)

            b_observations = observations[batch_inds]
            b_actions = actions[batch_inds]
            b_next_observations = observations[batch_inds + 1]
            b_rewards = rewards[batch_inds + 1]
            b_terminated = terminated[batch_inds + 1]
            weights = weights[batch_inds + 1]

            with torch.no_grad():
                target_max, _ = target_network(b_next_observations).max(dim=1)
            td_target = b_rewards + gamma * target_max * torch.logical_not(b_terminated).float()
            old_val = q_network(b_observations)[range(batch_size), b_actions]
            td_errors = (td_target - old_val) ** 2
            loss = torch.mean(td_errors)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update beta and priorities
            beta = min(1.0, beta + beta_increment_per_sampling)
            priorities[batch_inds] = td_errors

        # Update the target network
        if global_step % target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

env.close()
