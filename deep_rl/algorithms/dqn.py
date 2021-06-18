import torch

from deep_rl.common import MLP, rewards_to_returns
import numpy as np
import random
from collections import deque


class Dataset:
    def __init__(self):
        self.transitions = deque(maxlen=100000)

    def store(self, transition):
        self.transitions.append(transition)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.transitions))
        transitions = random.sample(self.transitions, batch_size)
        states = torch.FloatTensor([t[0] for t in transitions])
        actions = torch.LongTensor([t[1] for t in transitions]).unsqueeze(1)
        rewards = torch.FloatTensor([t[2] for t in transitions])
        next_states = torch.FloatTensor([t[3] for t in transitions])
        dones = torch.BoolTensor([t[4] for t in transitions])
        return states, actions, rewards, next_states, dones


class DQN:
    def __init__(self, env):
        """Implementation of DQN algorithm.

        Args:
            env (gym Env): The environment
            lr (float, optional): learning rate. Defaults to 0.001.
            epsilon_decay (float, otpional): epsilon_decay.
        """
        self.env = env

        self.dataset = Dataset()
        input_size = self.env.observation_space.shape[0]
        output_size = self.env.action_space.n
        self.Q_main = MLP([input_size, 64, 64, output_size])  # main
        self.Q_targ = MLP([input_size, 64, 64, output_size])  # target
        self.batch_size = 64
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.Q_main.parameters(), lr=0.01)

        self.epsilon_start = 0.9
        self.epsilon_decay = 0.99995
        self.epsilon_min = 0.05
        self.tau = 100
        self.gamma = 0.99

    def act(self, state, epsilon=0):
        """Espilon-greedy policy w.r.t. Q_main.

        Args:
            state (list): state as list.
            epsilon (float): probability of taking random action.
        """
        state = torch.FloatTensor(state)
        if random.random() > epsilon:  # epsilon greedy policy
            with torch.no_grad():  # the grad should not propagate into action choice
                values = self.Q_main(state)
                action = torch.argmax(values).item()
        else:
            action = self.env.action_space.sample()
        return action

    def train(self, nb_timesteps, test_fq=1000, nb_test_episodes=100, verbose=False):
        """Train the agent.

        Args:
            nb_timesteps (int): Number of training timesteps.
            test_fq (int): test every test_fq timesteps.
            nb_test_episodes (int): number of testing episodes.
        """
        # Start with epsilon = 1
        epsilon = self.epsilon_start
        test_returns = []
        done = True
        for timestep in range(nb_timesteps):
            if done:
                # reset and loop again
                state = self.env.reset()
                done = False
            # Sample episodes
            action = self.act(state, epsilon)
            next_state, reward, done, info = self.env.step(action)
            # store the transition
            self.dataset.store([state, action, reward, next_state, done])
            # next state becomes current state
            state = next_state

            # get minibatch from the dataset
            states, actions, rewards, next_states, dones = self.dataset.sample(
                self.batch_size
            )
            # target = r + γ max_a′(Q_targ(s′,a′)) if s' is non terminal
            # target = r + 0                       otherwise
            next_state_values = torch.max(self.Q_targ(next_states), dim=1)[0]
            next_state_values[dones] = 0
            target = rewards + self.gamma * next_state_values
            # prediction = Q(s′,a′)
            prediction = self.Q_main(states).gather(1, actions).squeeze(1)
            # loss = |target - prediction|
            loss = self.criterion(target, prediction)
            # Optimize the model : theta <- theta - grad_theta*loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if loss < 1.0:
            if timestep % self.tau == 0:
                # Q_targ <- Q_main
                self.Q_targ.load_state_dict(self.Q_main.state_dict())

            # decay epsilon
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

            # test and print mean return
            if timestep % test_fq == 0:
                test_return = self.test(nb_test_episodes)
                done = True  # testing function used env, we know need to reset
                if verbose:
                    print(
                        "Timestep: {0:6d}/{1:6d}    Mean test return: {2:6.2f}    Epsilon: {3:3.2f}    Loss: {4:4.2f}".format(
                            timestep, nb_timesteps, test_return, epsilon, loss
                        )
                    )
                test_returns.append(test_return)

        return test_returns

    def test(self, nb_test_episodes):
        """Test the policy nb_tests times and return the mean reward rate."""
        returns = []
        for _ in range(nb_test_episodes):
            state = self.env.reset()
            done = False
            rewards = []
            while not done:
                # interract
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                # store
                rewards.append(reward)
                state = next_state
                if done:
                    returns += rewards_to_returns(rewards, self.gamma)
        return np.mean(returns)
