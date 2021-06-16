import gym
from deep_rl.algorithms import REINFORCE


env = gym.make("CartPole-v1")
agent = REINFORCE(env, baseline_mean=False)

results = []
for _ in range(10):
    agent = REINFORCE(env, baseline_mean=False)
    result = agent.train(300000, 10000)
    results.append(result)
print(results)
