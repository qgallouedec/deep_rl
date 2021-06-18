import torch

from deep_rl.common import MLP, rewards_to_returns
import numpy as np


class REINFORCE:
    def __init__(self, env, baseline_mean=True):
        """Implementation of REINFORCE algorithm.

        Args:
            env (gym Env): The environment
            baseline_mean (bool, optional): Whether the mean return is used as baseline.
        """
        self.env = env
        self.nn = MLP([self.env.observation_space.shape[0], 4, self.env.action_space.n])
        self.optimizer = torch.optim.SGD(self.nn.parameters(), lr=0.005)
        self.baseline_mean = baseline_mean

        self.gamma = 0.99

    def act(self, state, deterministic=False):
        """Returns the action sample by the agent."""
        state = torch.FloatTensor(state)  # convert state into a torch.Tensor
        logits, distrib = self.predict(state)
        if deterministic:  # choose best action
            action = torch.argmax(logits)
        else:  # sample action
            action = distrib.sample()
        action = action.item()  # convert tensor into a float.
        return action

    def predict(self, state):
        """Return the probability distribution over the actions."""
        logits = self.nn(state)
        distrib = torch.distributions.Categorical(logits=logits)
        return logits, distrib

    def train(self, nb_timesteps, test_fq=1000, nb_test_episodes=100, verbose=False):
        test_returns = []
        state = self.env.reset()
        done = False
        states, actions, rewards = [], [], []
        for timestep in range(nb_timesteps):
            # interract
            action = self.act(state)
            next_state, reward, done, info = self.env.step(action)
            # store
            actions.append(action)
            states.append(state)
            rewards.append(reward)

            state = next_state

            if done:
                # learn
                returns = rewards_to_returns(rewards, self.gamma)
                states = torch.FloatTensor(states)
                actions = torch.IntTensor(actions)
                returns = torch.FloatTensor(returns)
                if self.baseline_mean:
                    returns = returns - returns.mean()
                _, distrib = self.predict(states) # pi(a | s)
                logprobs = distrib.log_prob(actions)  # = log(pi(a | s))
                # J = - G_t grad log(pi(a|s))
                loss = -(returns * logprobs).mean()  # J = - G_t grad log(pi(a|s))
                # Optimize the model : theta <- theta - grad J(theta)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # start new episode
                state = self.env.reset()
                states, actions, rewards = [], [], []
                done = False

            if timestep % test_fq == 0:
                test_return = self.test(nb_test_episodes)
                if verbose:
                    print(
                        "Timestep: {0:6d}/{1:6d}    Mean test return: {2:6.2f}".format(
                            timestep, nb_timesteps, test_return
                        )
                    )
                test_returns.append(test_return)
                # start new episode
                state = self.env.reset()
                states, actions, rewards = [], [], []
                done = False

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

