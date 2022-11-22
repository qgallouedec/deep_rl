import numpy as np


def rewards_to_returns(rewards, gamma=1):
    rewards = np.array(rewards, dtype=np.float)
    returns = np.copy(rewards)
    for _ in rewards:
        rewards = np.roll(rewards, -1)
        rewards[..., -1] = 0
        rewards *= gamma
        returns += rewards

    return list(returns)
