import gym
from deep_rl.algorithms import REINFORCE

results=[]
for _ in range(1):
    env = gym.make("CartPole-v1")
    agent = REINFORCE(env, baseline_mean=True)
    result = agent.train(600000, 10000, verbose=True)
    results.append(result)
