from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.distributions import Categorical, Distribution


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, observation: Tensor) -> Tensor:
        return self.critic(observation).squeeze(-1)

    def get_action_distribution(self, observation: Tensor) -> Distribution:
        logits = self.actor(observation)
        return Categorical(logits=logits)

    def get_action(self, observation: Tensor) -> Tuple[Tensor, Tensor]:
        distribution = self.get_action_distribution(observation)
        action = distribution.sample()
        return action, distribution.log_prob(action)


env_id = "CartPole-v1"

total_timesteps = 20_000
num_steps = 128
num_updates = total_timesteps // num_steps
minibatch_size = num_steps // 4
update_epochs = 4

gamma = 0.99
gae_lambda = 0.95
learning_rate = 2.5e-4
clip_coef = 0.2
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5

# Env setup
env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
env = TorchWrapper(env)

# Seeding
seed = 1
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Agent setup
agent = ActorCritic(env)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

# Storage setup (num_steps + 1 because we need the terminal values to compute the advantage)
observations = torch.zeros((num_steps + 1, *env.observation_space.shape))
values = torch.zeros((num_steps + 1))
actions = torch.zeros((num_steps + 1, *env.action_space.shape), dtype=torch.long)
log_probs = torch.zeros((num_steps + 1))
rewards = torch.zeros((num_steps + 1))
dones = torch.zeros((num_steps + 1))

# Init the env
observation = env.reset()
global_step = 0

# Loop
for update in range(num_updates):
    # Annealing the rate
    new_lr = (1.0 - update / num_updates) * learning_rate
    optimizer.param_groups[0]["lr"] = new_lr

    step = 0

    # Store initial
    observations[step] = observation
    with torch.no_grad():
        values[step] = agent.get_value(observation)

    while step < num_steps:
        # Compute action
        with torch.no_grad():
            action, log_prob = agent.get_action(observations[step])

        # Store
        actions[step] = action
        log_probs[step] = log_prob

        # Step
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
            print(f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}")

        # Update count
        step += 1
        global_step += 1

        # Store
        observations[step] = observation
        with torch.no_grad():
            values[step] = agent.get_value(observations[step])
        rewards[step] = reward
        dones[step] = done

    # Compute advanges and return
    advantages = torch.zeros_like(rewards)
    last_gae_lamda = 0
    for t in reversed(range(num_steps)):
        advantages[t] = (
            rewards[t + 1] + gamma * (1.0 - dones[t + 1]) * (values[t + 1] + gae_lambda * last_gae_lamda) - values[t]
        )
        last_gae_lamda = advantages[t]
    returns = advantages + values

    # Optimizing the policy and value network
    for epoch in range(update_epochs):
        b_inds = np.random.permutation(num_steps)
        for start in range(0, num_steps, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            b_observations = observations[mb_inds]
            b_values = values[mb_inds]
            b_actions = actions[mb_inds]
            b_log_probs = log_probs[mb_inds]
            b_returns = returns[mb_inds]
            b_advantages = advantages[mb_inds]

            action_distribution = agent.get_action_distribution(b_observations)

            # Policy loss
            b_advantages = (b_advantages - torch.mean(b_advantages)) / (torch.std(b_advantages) + 1e-8)  # norm advantages
            new_log_probs = action_distribution.log_prob(b_actions)
            ratio = torch.exp(new_log_probs - b_log_probs)
            pg_loss1 = -b_advantages * ratio
            pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

            # Entropy loss
            entropy_loss = torch.mean(action_distribution.entropy())

            # Clip V-loss
            new_values = agent.get_value(b_observations)
            v_loss_unclipped = (new_values - b_returns) ** 2
            v_clipped = b_values + torch.clamp(new_values - b_values, -clip_coef, clip_coef)
            v_loss_clipped = (v_clipped - b_returns) ** 2
            v_loss = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))

            # Total loss
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

    var_y = torch.var(values)
    explained_var = torch.nan if var_y == 0 else 1 - torch.var(values - returns) / var_y

env.close()
