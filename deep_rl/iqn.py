from typing import Any, Dict, Tuple

import gym
import numpy as np
import torch
import utils
from torch import Tensor, nn, optim


class TorchWrapper(gym.Wrapper):
    """
    Torch wrapper. Actions and observations are Tensors instead of arrays.
    """

    def step(self, action: Tensor) -> Tuple[Tensor, float, bool, Dict[str, Any]]:
        action = action.cpu().numpy()
        observation, reward, done, info = self.env.step(action)
        return torch.tensor(observation), reward, done, info

    def reset(self) -> Tensor:
        observation = self.env.reset()
        return torch.tensor(observation)


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
    def __init__(self, env, K=32, num_cosines=32, embedding_dim=7 * 7 * 64):
        super().__init__()
        # Feature extractor of DQN.
        self.dqn_net = nn.Sequential(
            nn.Conv2d(env.observation_space.shape[2], 32, kernel_size=8, stride=4),
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
        self.quantile_net = QuantileNetwork(num_actions=env.action_space.n)

        self.K = K

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


def evaluate_quantile_at_action(s_quantiles: Tensor, actions: Tensor) -> Tensor:
    """
    Retrieves the quantile values at the specified actions.

    Args:
        s_quantiles (Tensor): A tensor of shape (batch_size, N, num_quantiles) representing the quantiles for each state.
        actions (Tensor): A tensor of shape (batch_size, 1) representing the actions to evaluate the quantiles at.

    Returns:
        Tensor: A tensor of shape (batch_size, N, 1) representing the quantile values at the specified actions.
    """
    batch_size = s_quantiles.shape[0]
    N = s_quantiles.shape[1]

    # Expand actions into (batch_size, N, 1).
    action_index = actions[..., None, None].expand(batch_size, N, 1)

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


env_id = "PongNoFrameskip-v4"

total_timesteps = 10_000  # 50_000_000
learning_starts = 5_000  # 50_000

batch_size = 32
learning_rate = 5e-5
gamma = 0.99
train_frequency = 4
target_network_frequency = 10_000

# Env setup
env = TorchWrapper(utils.AtariWrapper(gym.make(env_id)))

# Create the agent and run.


N = 64
N_dash = 64
K = 32
num_cosines = 64
kappa = 1.0
memory_size = 1_000_000

start_e = 1
end_e = 0.01
exploration_fraction = 25
slope = (end_e - start_e) / (exploration_fraction * total_timesteps)

max_episode_steps = 27_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)


observations = torch.empty((memory_size, *env.observation_space.shape), dtype=torch.uint8)
next_observations = torch.empty((memory_size, *env.observation_space.shape), dtype=torch.uint8)
actions = torch.empty(memory_size, dtype=torch.long)
rewards = torch.empty(memory_size, dtype=torch.float32)
dones = torch.empty(memory_size, dtype=torch.float32)

global_step = 0
learning_steps = 0
episodes = 0

# Online network.
online_net = IQN(env, K=K, num_cosines=num_cosines).to(device)
# Target network.
target_net = IQN(env, K=K, num_cosines=num_cosines).to(device)

# Copy parameters of the learning network to the target network.
target_net.load_state_dict(online_net.state_dict())


optimizer = optim.Adam(online_net.parameters(), lr=learning_rate, eps=1e-2 / batch_size)


while True:
    episodes += 1
    episode_return = 0.0
    episode_steps = 0

    done = False
    observation = env.reset()

    while (not done) and episode_steps <= max_episode_steps:
        # Use e-greedy for evaluation.
        epsilon = max(slope * global_step + start_e, end_e)
        if global_step < learning_starts or np.random.rand() < epsilon:
            action = torch.tensor(env.action_space.sample())
        else:
            observation_ = observation.unsqueeze(0).to(device).float().permute(0, 3, 1, 2) / 255.0
            with torch.no_grad():
                embeddings = online_net.compute_embeddings(observation_)
                action = online_net.compute_q_values(embeddings).argmax()

        next_observation, reward, done, _ = env.step(action)

        # To calculate efficiently, I just set priority=max_priority here.
        observations[global_step % memory_size] = observation
        next_observations[global_step % memory_size] = next_observation
        actions[global_step % memory_size] = action
        rewards[global_step % memory_size] = reward
        dones[global_step % memory_size] = done

        global_step += 1
        episode_steps += 1
        episode_return += reward
        observation = next_observation

        if global_step % target_network_frequency == 0:
            target_net.load_state_dict(online_net.state_dict())

        if global_step % train_frequency == 0 and global_step >= learning_starts:
            learning_steps += 1
            batch_inds = np.random.randint(global_step, size=batch_size)
            b_observations = observations[batch_inds].to(device).float().permute(0, 3, 1, 2) / 255.0
            b_next_observations = next_observations[batch_inds].to(device).float().permute(0, 3, 1, 2) / 255.0
            b_actions = actions[batch_inds].to(device)
            b_rewards = rewards[batch_inds].to(device)
            b_dones = dones[batch_inds].to(device)

            # Compute the embeddings
            embeddings = online_net.compute_embeddings(b_observations)

            # Sample fractions
            taus = torch.rand(batch_size, N, dtype=embeddings.dtype, device=embeddings.device)
            # Calculate quantile values of current states and actions at tau_hats.
            quantiles = online_net.compute_quantiles(embeddings, taus)
            current_sa_quantiles = evaluate_quantile_at_action(quantiles, b_actions)  # (batch_size, N, 1)

            with torch.no_grad():
                # Calculate Q values of next states.
                next_embeddings = target_net.compute_embeddings(b_next_observations)
                next_q_value = target_net.compute_q_values(next_embeddings)

                # Calculate greedy actions.
                next_actions = torch.argmax(next_q_value, dim=1)  # (batch_size,)

                # Sample next fractions.
                tau_dashes = torch.rand(batch_size, N_dash, dtype=embeddings.dtype, device=embeddings.device)

                # Calculate quantile values of next states and next actions.
                target_quantile = target_net.compute_quantiles(next_embeddings, tau_dashes)
                next_sa_quantiles = evaluate_quantile_at_action(target_quantile, next_actions)
                next_sa_quantiles = next_sa_quantiles.transpose(1, 2)  # (batch_size, 1, N_dash)

                # Calculate target quantile values.
                target_sa_quantiles = (
                    b_rewards[..., None, None] + (1.0 - b_dones[..., None, None]) * gamma * next_sa_quantiles
                )  # (batch_size, 1, N_dash)

            td_errors = target_sa_quantiles - current_sa_quantiles
            quantile_loss = calculate_quantile_huber_loss(td_errors, taus, kappa)

            optimizer.zero_grad()
            quantile_loss.backward()
            optimizer.step()

    print(f"Episode: {episodes:<4}  " f"episode steps: {episode_steps:<4}  " f"return: {episode_return:<5.1f}")
    if global_step > total_timesteps:
        break

env.close()
