import gym
from deep_rl.algorithms import DQN

losses = []
for _ in range(20):
    env = gym.make("CartPole-v1")
    agent = DQN(env)
    losses.append(agent.train(600, 10, verbose=False))

print(losses)
# for _ in range(100):
#     state = env.reset()
#     done = False
#     while not done:
#         action = agent.act(state)
#         state, _, done, _ = env.step(action)
#         env.render()