import torch
from torch import optim
from torch.distributions import Categorical

from deep_rl.common import MLP, rewards_to_returns
import numpy as np


class REINFORCE:
    def __init__(self, env, lr=0.001, baseline_mean=True):
        """Implementation of REINFORCE algorithm.

        Args:
            env (gym Env): The environment
            lr (float, optional): learning rate. Defaults to 0.001.
            baseline_mean (bool, optional): Whether the mean return is used as baseline.
        """
        self.env = env
        self.nn = MLP([self.env.observation_space.shape[0], 4, self.env.action_space.n])
        self.optimizer = optim.SGD(self.nn.parameters(), lr=lr)
        self.baseline_mean = baseline_mean

    def act(self, state, deterministic=False):
        """Returns the action sample by the agent."""
        distrib = self.predict(state)
        action = distrib.sample().item()
        return action

    def predict(self, state):
        """Return the distribution ober the actions"""
        state = (
            torch.tensor(state, dtype=torch.float32)
            if not torch.is_tensor(state)
            else state
        )
        logits = self.nn(state)
        distrib = Categorical(logits=logits)
        return distrib

    def learn(self, states, actions, returns):
        # Calculate loss
        states = torch.FloatTensor(states)
        actions = torch.IntTensor(actions)
        returns = torch.FloatTensor(returns)
        if self.baseline_mean:
            returns = returns - returns.mean()
        logprobs = self.predict(states).log_prob(actions)  # = log(pi(a | s))
        loss = -(returns * logprobs).mean()  # J = - G_t grad log(pi(a|s))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_timesteps):
        """Train the agent for num_timesteps."""
        timesteps_left = num_timesteps
        state = self.env.reset()
        done = False
        states, actions, rewards = [], [], []
        while timesteps_left > 0:
            # interract
            action = self.act(state)
            next_state, reward, done, info = self.env.step(action)
            # store
            actions.append(action)
            states.append(state)
            rewards.append(reward)

            state = next_state
            timesteps_left -= 1

            if done or timesteps_left == 0:
                returns = rewards_to_returns(rewards)
                self.learn(states, actions, returns)
                state = self.env.reset()
                done = False
                states, actions, rewards = [], [], []

    def test(self, nb_tests):
        """Test the policy nb_tests times and return the mean reward rate."""
        sum_rewards = []
        for _ in range(nb_tests):
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
                    sum_rewards.append(np.sum(rewards))

        return np.mean(sum_rewards)

    def play_once(self):
        """Play once with rendering"""
        state = self.env.reset()
        done = False
        while not done:
            action = self.act(state, deterministic=True)
            state, _, done, _ = self.env.step(action)
            self.env.render()
