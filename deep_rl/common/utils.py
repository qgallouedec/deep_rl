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


if __name__ == "__main__":
    rewards = [[1.0, 2, 3], [3, 4, 5]]
    print(rewards_to_returns(rewards, 0.5))
