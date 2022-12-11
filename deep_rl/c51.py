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
    def __init__(self, env, n_atoms=101):
        super().__init__()
        self.n_atoms = n_atoms
        self.n = env.action_space.n
        self.network = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_atoms),
            nn.Unflatten(-1, (self.n, self.n_atoms)),
        )

    def get_probs(self, observation: Tensor) -> Tensor:
        return torch.softmax(self.network(observation), dim=-1)


env_id = "CartPole-v1"

total_timesteps = 20_000
learning_starts = 10_000

start_e = 1
end_e = 0.05
exploration_fraction = 0.5
slope = (end_e - start_e) / (exploration_fraction * total_timesteps)

train_frequency = 10
batch_size = 128
gamma = 0.99
learning_rate = 2.5e-4
target_network_frequency = 500

v_min = -100
v_max = 100
n_atoms = 101
delta_z = (v_max - v_min) / (n_atoms - 1)
atoms = torch.linspace(v_min, v_max, steps=n_atoms)

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
q_network = QNetwork(env, n_atoms=n_atoms)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate, eps=0.01 / batch_size)
target_network = QNetwork(env, n_atoms=n_atoms)
target_network.load_state_dict(q_network.state_dict())

# Storage setup
observations = torch.zeros((total_timesteps + 1, *env.observation_space.shape))
actions = torch.zeros((total_timesteps + 1, *env.action_space.shape), dtype=torch.long)
rewards = torch.zeros((total_timesteps + 1))
terminated = torch.zeros((total_timesteps + 1), dtype=torch.bool)

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
        probs = q_network.get_probs(observation)
        q_values = torch.sum(probs * atoms, dim=-1)
        action = torch.argmax(q_values, -1)

    # Store
    actions[global_step] = action

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
        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

    # Optimize the agent
    if global_step >= learning_starts:
        if global_step % train_frequency == 0:
            batch_inds = np.random.randint(global_step, size=batch_size)

            b_observations = observations[batch_inds]
            b_actions = actions[batch_inds]
            b_next_observations = observations[batch_inds + 1]
            b_rewards = rewards[batch_inds + 1]
            b_terminated = terminated[batch_inds + 1]

            next_atoms = b_rewards.unsqueeze(1) + gamma * atoms * torch.logical_not(b_terminated).unsqueeze(1).float()

            # Projection
            tz = torch.clamp(next_atoms, v_min, v_max)
            b = (tz - v_min) / delta_z
            l = b.floor()
            u = b.ceil()

            with torch.no_grad():
                probs = target_network.get_probs(b_next_observations)

            q_values = torch.sum(probs * atoms, dim=-1)
            action = torch.argmax(q_values, -1)
            next_probs = probs[torch.arange(action.shape[0]), action]

            # (l == u).float() handles the case where bj is exactly an integer
            # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
            d_m_l = (u + (l == u).float() - b) * next_probs
            d_m_u = (b - l) * next_probs
            target_probs = torch.zeros_like(next_probs)
            for i in range(target_probs.size(0)):
                target_probs[i].index_add_(0, l.long()[i], d_m_l[i])  # ml ← ml +pj(xt+1,a∗)(u − bj)
                target_probs[i].index_add_(0, u.long()[i], d_m_u[i])  # mu ← mu +pj(xt+1,a∗)(bj − l)

            probs = q_network.get_probs(b_observations)[torch.arange(batch_size), b_actions]

            loss = torch.mean(-torch.sum((target_probs * torch.log(probs + 1e-8)), dim=-1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the target network
        if global_step % target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

env.close()
