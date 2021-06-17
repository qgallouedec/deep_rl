import gym
from deep_rl.algorithms import DQN


env = gym.make("CartPole-v1")
results = []
for _ in range(20):
    agent = DQN(env)
    results.append(agent.train(3000000, 10000))

print(results)
