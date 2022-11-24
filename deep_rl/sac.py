from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
from torch import Tensor, nn, optim
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -5


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


class SoftQNetwork(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape) + np.prod(env.action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([observation, action], dim=1)
        value = self.network(x).squeeze(1)
        return value


class Actor(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_net = nn.Linear(256, np.prod(env.action_space.shape))
        self.log_std_net = nn.Sequential(
            nn.Linear(256, np.prod(env.action_space.shape)),
            nn.Tanh(),
        )
        action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, observation: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.shared_net(observation)
        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, observation: Tensor) -> Tuple[Tensor, Tensor]:
        mean, log_std = self.forward(observation)
        distribution = Normal(mean, log_std.exp())
        x_t = distribution.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = distribution.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1)
        return action, log_prob


env_id = "HopperBulletEnv-v0"

total_timesteps = 6_000
learning_starts = 5_000

policy_frequency = 2
batch_size = 256
target_network_frequency = 1
gamma = 0.99
tau = 0.005
policy_lr = 3e-4
q_lr = 1e-3
alpha_lr = q_lr

# Env setup
env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
env = TorchWrapper(env)

# Seeding
seed = 1
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
env.action_space.seed(seed)

# Actor setup
actor = Actor(env)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=policy_lr)

# Networks setup
qf1 = SoftQNetwork(env)
qf2 = SoftQNetwork(env)
qf1_target = SoftQNetwork(env)
qf2_target = SoftQNetwork(env)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=q_lr)

target_entropy = -torch.prod(torch.Tensor(env.action_space.shape)).item()
log_alpha = torch.zeros(1, requires_grad=True)
alpha = log_alpha.exp().item()
alpha_optimizer = optim.Adam([log_alpha], lr=alpha_lr)


# Storage setup
observations = torch.zeros((total_timesteps + 1, *env.observation_space.shape))
actions = torch.zeros((total_timesteps + 1, *env.action_space.shape))
rewards = torch.zeros((total_timesteps + 1))
terminated = torch.zeros((total_timesteps + 1), dtype=torch.bool)

# Initiate the envrionment and store the inital observation
observation = env.reset()
global_step = 0
observations[global_step] = observation

# Loop
while global_step < total_timesteps:
    if global_step < learning_starts:
        action = torch.tensor(env.action_space.sample())
    else:
        with torch.no_grad():
            action, _ = actor.get_action(observation)

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
        print(f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}")

    # Optimize target and agent
    if global_step >= learning_starts:
        batch_inds = np.random.randint(global_step, size=batch_size)

        b_observations = observations[batch_inds]
        b_actions = actions[batch_inds]
        b_next_observations = observations[batch_inds + 1]
        b_rewards = rewards[batch_inds + 1]
        b_terminated = terminated[batch_inds + 1]

        with torch.no_grad():
            next_pred_actions, next_pred_log_probs = actor.get_action(b_next_observations)
            qf1_next_targets = qf1_target(b_next_observations, next_pred_actions)
            qf2_next_targets = qf2_target(b_next_observations, next_pred_actions)

        min_qf_next_targets = torch.min(qf1_next_targets, qf2_next_targets) - alpha * next_pred_log_probs
        next_q_values = b_rewards + torch.logical_not(b_terminated).float() * gamma * min_qf_next_targets

        qf1_values = qf1(b_observations, b_actions)
        qf2_values = qf2(b_observations, b_actions)
        qf1_loss = torch.mean((qf1_values - next_q_values) ** 2)
        qf2_loss = torch.mean((qf2_values - next_q_values) ** 2)
        qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        if global_step % policy_frequency == 0:  # TD 3 Delayed update support
            for _ in range(policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pred_actions, log_probs = actor.get_action(b_observations)
                qf1_values = qf1(b_observations, pred_actions)
                qf2_values = qf2(b_observations, pred_actions)
                min_qf_pi = torch.min(qf1_values, qf2_values)
                actor_loss = torch.mean((alpha * log_probs) - min_qf_pi)

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                with torch.no_grad():
                    _, log_probs = actor.get_action(b_observations)
                alpha_loss = torch.mean(-log_alpha * (log_probs + target_entropy))

                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha = log_alpha.exp().item()

        # Update the target networks
        if global_step % target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

env.close()
