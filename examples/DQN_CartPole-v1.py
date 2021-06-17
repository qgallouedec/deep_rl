import gym
from deep_rl.algorithms import DQN


for _ in range(20):
    env = gym.make("CartPole-v1")
    agent = DQN(env)
    print(agent.train(60000, 1000, verbose=False))

# for _ in range(100):
#     state = env.reset()
#     done = False
#     while not done:
#         action = agent.act(state)
#         state, _, done, _ = env.step(action)
#         env.render()