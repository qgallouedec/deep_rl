import torch
from torch import optim
from torch.distributions import Categorical

from deep_rl.common import MLP, rewards_to_returns
from deep_rl.common.mpi import gather
import numpy as np
import random
from collections import deque

from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()


class Dataset:
    def __init__(self):
        self.transitions = deque(maxlen=100000)

    def store(self, transition):
        transitions = gather([transition])
        # = [[s1, a1, r1, s1'], [s2, a2, r2, s2'], ...]
        if RANK == 0:  # only store in root
            self.transitions += transitions

    def sample(self, batch_size):
        transitions = random.sample(self.transitions, batch_size)
        states = [transition[0] for transition in transitions]
        actions = [transition[1] for transition in transitions]
        rewards = [transition[2] for transition in transitions]
        next_states = [transition[3] for transition in transitions]
        return states, actions, rewards, next_states


class DQN:
    def __init__(self, env, lr=0.01, epsilon_decay=0.99, tau=10):
        """Implementation of DQN algorithm.

        Args:
            env (gym Env): The environment
            lr (float, optional): learning rate. Defaults to 0.001.
            epsilon_decay (float, otpional): epsilon_decay.
        """
        self.env = env
        self.main_Q = MLP(
            [self.env.observation_space.shape[0], 4, self.env.action_space.n]
        )
        self.target_Q = MLP(
            [self.env.observation_space.shape[0], 4, self.env.action_space.n]
        )
        self._is_mpi_root = RANK == 0
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.gamma = 0.99
        self.dataset = Dataset()

        # Since the leanring is only on root, no need
        # to define optimizer and baseline_mean on other workers
        if self._is_mpi_root:
            self.criterion = torch.nn.L1Loss()
            self.optimizer = optim.SGD(self.main_Q.parameters(), lr=lr)

    def act(self, state, epsilon=0):
        """Returns the action sample by the agent."""
        state = torch.FloatTensor(state)
        if random.random() > epsilon:  # epsilon greedy policy
            values = self.main_Q(state)
            action = torch.argmax(values).item()
        else:
            action = self.env.action_space.sample()
        return action

    def sample_episodes(self, num_timesteps, epsilon):
        """Sample episode and store them inn the dataset."""
        timesteps_left = num_timesteps
        state = self.env.reset()
        done = False
        while timesteps_left > 0:
            # interract
            action = self.act(state, epsilon)
            next_state, reward, done, info = self.env.step(action)
            # store
            self.dataset.store([state, action, reward, next_state])

            state = next_state
            timesteps_left -= 1

            if done:
                # reset and loop again
                state = self.env.reset()
                done = False

    def train(self, nb_timesteps, test_fq, nb_test_rollout=100):
        nb_epochs = nb_timesteps // test_fq
        nb_timesteps_per_epoch = nb_timesteps // nb_epochs
        # share the training in all workers
        nb_timesteps_per_rollout = nb_timesteps_per_epoch // SIZE
        epsilon = 1
        success_rates = []
        for epoch in range(nb_epochs):
            self.sample_episodes(nb_timesteps_per_rollout, epsilon)
            for _ in range(1000):
                if self._is_mpi_root:  # perform learning in root worker
                    states, actions, rewards, next_states = self.dataset.sample(256)
                    states = torch.FloatTensor(states)
                    actions = torch.tensor(actions)
                    rewards = torch.IntTensor(rewards)
                    next_states = torch.FloatTensor(next_states)
                    # target = r + γ max_a′(Q_targ(s′,a′))
                    target = rewards + self.gamma * torch.max(
                        self.target_Q(next_states)
                    )
                    # loss = target - Q(s′,a′)  (stop grad of target)
                    loss = self.criterion(
                        target.detach(),
                        torch.index_select(self.main_Q(states), 1, actions),
                    )
                    # Optimize the model : theta <- theta - grad theta
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    # broadcast the network to update every workers
                    MPI.COMM_WORLD.bcast(self.main_Q.state_dict())
                else:  # on non-root workers
                    # recieved and update network
                    state_dict = MPI.COMM_WORLD.bcast(None)
                    self.main_Q.load_state_dict(state_dict)
            if epoch % self.tau == 0:
                self.target_Q.load_state_dict(self.main_Q.state_dict())
            epsilon *= self.epsilon_decay

            if self._is_mpi_root:  # do not execute test on non-root worker
                success_rate = self.test(nb_test_rollout)
                print(epoch, success_rate, epsilon)
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
            action = self.act(state, epsilon=0)
            state, _, done, _ = self.env.step(action)
            self.env.render()
