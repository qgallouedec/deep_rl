import gym
from deep_rl.algorithms import REINFORCE

env = gym.make("CartPole-v1")
agent = REINFORCE(env, baseline_mean=False)


for t in range(100):
    print(t * 10000, agent.test(1000))
    agent.train(10000)
    agent.play_once()
print(100*10000, agent.test(1000))
