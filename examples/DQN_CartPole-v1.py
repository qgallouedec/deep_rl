import gym

from deep_rl.algorithms import DQN
from deep_rl.utils import results_to_dat

results = []
for _ in range(10):
    env = gym.make("CartPole-v1")
    agent = DQN(env)
    timesteps, result = agent.train(100000, 1000, verbose=False)
    results.append(result)

results_to_dat(timesteps, results)
