from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.distributions import Categorical

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


env_id = "CartPole-v1"

gamma = 0.99

# Env setup
env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
env = TorchWrapper(env)

# Seeding
seed = 1
env.seed(seed)
torch.manual_seed(seed)

agent = nn.Sequential(
    nn.Linear(4, 128),
    nn.Dropout(p=0.6),
    nn.ReLU(),
    nn.Linear(128, 2),
    nn.Softmax(-1),
)
optimizer = optim.Adam(agent.parameters(), lr=1e-2)

global_step = 0

for episode_idx in range(100):

    log_probs = torch.zeros((env.spec.max_episode_steps + 1))
    returns = torch.zeros((env.spec.max_episode_steps + 1))

    observation = env.reset()
    done = False
    step = 0
    while not done:
        probs = agent(observation)
        distribution = Categorical(probs)
        action = distribution.sample()
        log_probs[step] = distribution.log_prob(action)
        observation, reward, done, info = env.step(action)
        step += 1
        global_step += 1
        returns[:step] += gamma ** torch.flip(torch.arange(step), (0,)) * reward

    print(f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}")

    b_returns = returns[:step]
    b_log_probs = log_probs[:step]
    b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + np.exp(LOG_STD_MIN))
    policy_loss = torch.sum(-b_log_probs * b_returns)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
