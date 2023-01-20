import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam

import utils


class LazyMultiStepMemory(dict):
    state_keys = ["state", "next_state"]
    np_keys = ["action", "reward", "done"]
    keys = state_keys + np_keys

    def __init__(self, capacity, state_shape, device):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.device = device
        self["state"] = []
        self["next_state"] = []
        self["action"] = np.empty((self.capacity, 1), dtype=np.int64)
        self["reward"] = np.empty((self.capacity, 1), dtype=np.float32)
        self["done"] = np.empty((self.capacity, 1), dtype=np.float32)

        self._n = 0
        self._p = 0

    def append(self, state, action, reward, next_state, done):
        self["state"].append(state)
        self["next_state"].append(next_state)
        self["action"][self._p] = action
        self["reward"][self._p] = reward
        self["done"][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity
        # Truncate
        while len(self["state"]) > self.capacity:
            del self["state"][0]
            del self["next_state"][0]

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self["state"]), size=batch_size)
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty((batch_size, *self.state_shape), dtype=np.uint8)
        next_states = np.empty((batch_size, *self.state_shape), dtype=np.uint8)

        for i, index in enumerate(indices):
            _index = np.mod(index + bias, self.capacity)
            states[i, ...] = self["state"][_index]
            next_states[i, ...] = self["next_state"][_index]

        states = torch.ByteTensor(states).to(self.device).float().permute(0, 3, 1, 2) / 255.0
        next_states = torch.ByteTensor(next_states).to(self.device).float().permute(0, 3, 1, 2) / 255.0
        actions = torch.LongTensor(self["action"][indices]).to(self.device)
        rewards = torch.FloatTensor(self["reward"][indices]).to(self.device)
        dones = torch.FloatTensor(self["done"][indices]).to(self.device)

        return states, actions, rewards, next_states, dones


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class CosineEmbeddingNetwork(nn.Module):
    """
    Computes the embeddings of tau values using cosine functions.

    Args:
        num_cosines (int, optional): Number of cosines to use for embedding. Default is 64.
        embedding_dim (int, optional): Dimension of the embedding. Default is 7 * 7 * 64.
    """

    def __init__(self, num_cosines: int = 64, embedding_dim=7 * 7 * 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_cosines, embedding_dim),
            nn.ReLU(),
        )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus: Tensor) -> Tensor:
        """
        Calculate the embeddings of tau values.

        Args:
            taus (Tensor): A tensor of shape (batch_size, N) representing the tau values.

        Returns:
            Tensor: A tensor of shape (batch_size, N, embedding_dim) representing the embeddings of tau values.
        """
        N = taus.shape[1]

        # Compute [pi, 2*pi, 3*pi, 4*pi, ..., num_cosines*pi]
        i_pi = np.pi * torch.arange(start=1, end=self.num_cosines + 1, dtype=taus.dtype, device=taus.device)
        i_pi = i_pi.reshape(1, 1, self.num_cosines)  # [1, 1, num_cosines]

        # Compute cos(i * pi * tau)
        taus = torch.unsqueeze(taus, dim=-1)  # [batch_size, N, 1]
        cosines = torch.cos(taus * i_pi)  # [batch_size, N, num_cosines]
        cosines = torch.flatten(cosines, end_dim=1)  # [batch_size * N, num_cosines]

        # Compute embeddings of taus
        tau_embeddings = self.net(cosines)  # [batch_size * N, embedding_dim]
        tau_embeddings = torch.reshape(tau_embeddings, (-1, N, self.embedding_dim))  # [batch_size, N, embedding_dim]
        return tau_embeddings


class QuantileNetwork(nn.Module):
    """
    Compute the quantile values of actions given the embeddings and tau values.

    Args:
        num_actions (int): Number of actions.
        embedding_dim (int, optional): Dimension of the embeddings. Default is 7 * 7 * 64.
    """

    def __init__(self, num_actions: int, embedding_dim: int = 7 * 7 * 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

    def forward(self, embeddings: Tensor, tau_embeddings: Tensor) -> Tensor:
        """
        Compute the quantile values of actions given the embeddings and tau values.

        Args:
            - embeddings (Tensor): Tensor of shape (batch_size, embedding_dim) representing the embeddings.
            - tau_embeddings (Tensor): Tensor of shape (batch_size, N, embedding_dim) representing of tau embeddings values.

        Returns:
            - Tensor: A tensor of shape (batch_size, N, num_actions) representing the quantile values of actions.
        """
        N = tau_embeddings.shape[1]

        # Compute the embeddings and taus
        embeddings = torch.unsqueeze(embeddings, dim=1)  # [batch_size, 1, self.embedding_dim]
        embeddings = embeddings * tau_embeddings  # [batch_size, N, self.embedding_dim]

        # Compute the quantile values
        embeddings = torch.flatten(embeddings, end_dim=1)  # [batch_size * N, self.embedding_dim]
        quantiles = self.net(embeddings)
        quantiles = torch.reshape(quantiles, shape=(-1, N, self.num_actions))
        return quantiles


class IQN(nn.Module):
    def __init__(self, num_channels, num_actions, K=32, num_cosines=32, embedding_dim=7 * 7 * 64):
        super().__init__()

        # Feature extractor of DQN.
        self.dqn_net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        ).apply(initialize_weights_he)
        # Cosine embedding network.
        self.cosine_net = CosineEmbeddingNetwork(num_cosines=num_cosines, embedding_dim=embedding_dim)
        # Quantile network.
        self.quantile_net = QuantileNetwork(num_actions=num_actions)

        self.K = K
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def compute_embeddings(self, observations: Tensor) -> Tensor:
        return self.dqn_net(observations)

    def compute_quantiles(self, embeddings: Tensor, taus: Tensor) -> Tensor:
        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(embeddings, tau_embeddings)

    def compute_q_values(self, embeddings: Tensor) -> Tensor:
        batch_size = embeddings.shape[0]

        # Sample fractions
        taus = torch.rand(batch_size, self.K, dtype=embeddings.dtype, device=embeddings.device)

        # Compute quantiles
        quantiles = self.compute_quantiles(embeddings, taus)  # (batch_size, K, num_actions)

        # Calculate expectations of value distributions.
        q_values = torch.mean(quantiles, dim=1)  # (batch_size, num_actions)
        return q_values


def evaluate_quantile_at_action(s_quantiles, actions):
    assert s_quantiles.shape[0] == actions.shape[0]

    batch_size = s_quantiles.shape[0]
    N = s_quantiles.shape[1]

    # Expand actions into (batch_size, N, 1).
    action_index = actions[..., None].expand(batch_size, N, 1)

    # Calculate quantile values at specified actions.
    sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

    return sa_quantiles


def calculate_quantile_huber_loss(td_errors, taus, kappa):
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = torch.where(
        td_errors.abs() <= kappa, 0.5 * td_errors.pow(2), kappa * (td_errors.abs() - 0.5 * kappa)
    )
    assert element_wise_huber_loss.shape == (batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = (
        torch.abs(taus[..., None] - (td_errors.detach() < 0).float()) * element_wise_huber_loss / kappa
    )
    assert element_wise_quantile_huber_loss.shape == (batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)

    quantile_huber_loss = batch_quantile_huber_loss.mean()

    return quantile_huber_loss


import gym

env_id = "PongNoFrameskip-v4"

# Create environments.
env = utils.AtariWrapper(gym.make(env_id))

# Create the agent and run.

num_steps = 10_000  # 50_000_000
batch_size = 32
N = 64
N_dash = 64
K = 32
num_cosines = 64
kappa = 1.0
lr = 5e-5
memory_size = 1_000_000
gamma = 0.99
update_interval = 4
target_update_interval = 10_000
start_steps = 5_000  # 50_000
epsilon_train = 0.01
epsilon_decay_steps = 250_000
log_interval = 100
max_episode_steps = 27_000
cuda = True
seed = 0


torch.manual_seed(seed)
np.random.seed(seed)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

# Replay memory which is memory-efficient to store stacked frames.
memory = LazyMultiStepMemory(memory_size, env.observation_space.shape, device)

steps = 0
learning_steps = 0
episodes = 0
num_actions = env.action_space.n
num_steps = num_steps
batch_size = batch_size

# Online network.
online_net = IQN(num_channels=env.observation_space.shape[2], num_actions=num_actions, K=K, num_cosines=num_cosines).to(device)
# Target network.
target_net = IQN(num_channels=env.observation_space.shape[2], num_actions=num_actions, K=K, num_cosines=num_cosines).to(device)

# Copy parameters of the learning network to the target network.
target_net.load_state_dict(online_net.state_dict())
# Disable calculations of gradients of the target network.
for param in target_net.parameters():
    param.requires_grad = False

optim = Adam(online_net.parameters(), lr=lr, eps=1e-2 / batch_size)


while True:
    online_net.train()
    target_net.train()

    episodes += 1
    episode_return = 0.0
    episode_steps = 0

    done = False
    observation = env.reset()

    while (not done) and episode_steps <= max_episode_steps:
        # Use e-greedy for evaluation.
        epsilon = max(epsilon_train, 1 - (1 - epsilon_train) / epsilon_decay_steps * steps)
        if steps < start_steps or np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            observation_ = torch.ByteTensor(observation).unsqueeze(0).to(device).float().permute(0, 3, 1, 2) / 255.0
            with torch.no_grad():
                embeddings = online_net.compute_embeddings(observation_)
                action = online_net.compute_q_values(embeddings).argmax().item()

        next_observation, reward, done, _ = env.step(action)

        # To calculate efficiently, I just set priority=max_priority here.
        memory.append(observation, action, reward, next_observation, done)

        steps += 1
        episode_steps += 1
        episode_return += reward
        observation = next_observation

        if steps % target_update_interval == 0:
            target_net.load_state_dict(online_net.state_dict())

        if steps % update_interval == 0 and steps >= start_steps:
            learning_steps += 1
            observations, actions, rewards, next_observations, dones = memory.sample(batch_size)

            # Compute the embeddings
            embeddings = online_net.compute_embeddings(observations)

            # Sample fractions.
            taus = torch.rand(batch_size, N, dtype=embeddings.dtype, device=embeddings.device)
            # Calculate quantile values of current states and actions at tau_hats.
            current_sa_quantiles = evaluate_quantile_at_action(
                online_net.compute_quantiles(embeddings, taus), actions
            )  # (batch_size, N, 1)

            with torch.no_grad():
                # Calculate Q values of next states.
                next_embeddings = target_net.compute_embeddings(next_observations)
                next_q_value = target_net.compute_q_values(next_embeddings)

                # Calculate greedy actions.
                next_actions = torch.argmax(next_q_value, dim=1, keepdim=True)  # (batch_size, 1)

                # Sample next fractions.
                tau_dashes = torch.rand(batch_size, N_dash, dtype=embeddings.dtype, device=embeddings.device)

                # Calculate quantile values of next states and next actions.
                next_sa_quantiles = evaluate_quantile_at_action(
                    target_net.compute_quantiles(next_embeddings, tau_dashes), next_actions
                ).transpose(
                    1, 2
                )  # (batch_size, 1, N_dash)

                # Calculate target quantile values.
                target_sa_quantiles = (
                    rewards[..., None] + (1.0 - dones[..., None]) * gamma * next_sa_quantiles
                )  # (batch_size, 1, N_dash)

            td_errors = target_sa_quantiles - current_sa_quantiles
            quantile_loss = calculate_quantile_huber_loss(td_errors, taus, kappa)

            optim.zero_grad()
            quantile_loss.backward()
            optim.step()

    print(f"Episode: {episodes:<4}  " f"episode steps: {episode_steps:<4}  " f"return: {episode_return:<5.1f}")
    if steps > num_steps:
        break

env.close()
