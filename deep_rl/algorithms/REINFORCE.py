import torch
from torch import optim
from torch.distributions import Categorical

from deep_rl.common import MLP, rewards_to_returns
from deep_rl.common.mpi import gather
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()

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
        self._is_mpi_root = comm.Get_rank() == 0

        # Since the leanring is only on root, no need
        # to define optimizer and baseline_mean on other workers
        if self._is_mpi_root:
            self.optimizer = optim.SGD(self.nn.parameters(), lr=lr)
            self.baseline_mean = baseline_mean

    def act(self, state, deterministic=False):
        """Returns the action sample by the agent."""
        distrib = self.predict(state)
        action = distrib.sample().item() # sample action and convert into a single float.
        return action

    def predict(self, state):
        """Return the distribution over the actions."""
        state = (
            torch.tensor(state, dtype=torch.float32)
            if not torch.is_tensor(state)
            else state
        )
        logits = self.nn(state)
        distrib = Categorical(logits=logits)
        return distrib

    def learn(self, states, actions, returns):
        """Learn from a batch of states, action and returns."""
        # gather the experience from all mpi workers
        states = gather(states)
        actions = gather(actions)
        returns = gather(returns)
        # perform learning in root worker
        if self._is_mpi_root:
            states = torch.FloatTensor(states)
            actions = torch.IntTensor(actions)
            returns = torch.FloatTensor(returns)

            if self.baseline_mean:
                returns = returns - returns.mean()
            logprobs = self.predict(states).log_prob(actions)  # = log(pi(a | s))
            loss = -(returns * logprobs).mean()  # J = - G_t grad log(pi(a|s))
            # Optimize the model : theta <- theta - grad theta
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
            MPI.COMM_WORLD.bcast(self.nn.state_dict())
        else:
            state_dict = MPI.COMM_WORLD.bcast(None)
            self.nn.load_state_dict(state_dict)

    def train_once(self, num_timesteps):
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
                # learn
                returns = rewards_to_returns(rewards)
                self.learn(states, actions, returns)
                # reset and loop again
                state = self.env.reset()
                done = False
                states, actions, rewards = [], [], []
    
    def train(self, nb_timesteps, test_fq, nb_test_rollout=1000):
        nb_epochs = nb_timesteps // test_fq # test model every nb_rollout timesteps
        nb_timesteps_per_epoch = nb_timesteps // nb_epochs
        nb_timesteps_per_rollout = nb_timesteps_per_epoch // SIZE # share the training in all workers
        success_rates = []
        for epoch in range(nb_epochs):
            if self._is_mpi_root: # do not execute test on non-root worker
                success_rate = self.test(nb_test_rollout)
                print(epoch, success_rate)
                success_rates.append(success_rate)
                self.play_once()
            self.train_once(nb_timesteps_per_rollout)
            
        if self._is_mpi_root: # do not execute test on non-root worker
            success_rate = self.test(nb_test_rollout)
            print(epoch, success_rate)
            success_rates.append(success_rate)
    
        return success_rates

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
