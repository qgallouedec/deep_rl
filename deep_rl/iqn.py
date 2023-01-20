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
        Compute the embeddings of tau values.

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

    def compute_quantile_at_action(self, embeddings: Tensor, taus: Tensor, actions: Tensor) -> Tensor:
        """
        Retrieves the quantile values at the specified actions.

        Args:
            s_quantiles (Tensor): A tensor of shape (batch_size, N, num_quantiles) representing the quantiles for each observation.
            actions (Tensor): A tensor of shape (batch_size,) representing the actions to evaluate the quantiles at.

        Returns:
            Tensor: A tensor of shape (batch_size, N) representing the quantile values at the specified actions.
        """
        # Compute quantiles
        tau_embeddings = self.cosine_net(taus)
        s_quantiles = self.quantile_net(embeddings, tau_embeddings)

        # Copy actions N times to get a tensor a shape (batch_size, N)
        action_index = actions[..., None].expand(-1, s_quantiles.shape[1])

        # Compute quantile values at specified actions. The notation seems eavy notation,
        # but just select value of s_quantile (B, N, num_quantiles) with action_indexes (B, K).
        # Output shape is thus (B, K)
        sa_quantiles = s_quantiles.gather(dim=2, index=action_index.unsqueeze(-1)).squeeze(-1)
        return sa_quantiles

    def compute_q_values(self, embeddings: Tensor) -> Tensor:
        batch_size = embeddings.shape[0]

        # Sample fractions
        taus = torch.rand(batch_size, self.K, dtype=embeddings.dtype, device=embeddings.device)

        # Compute quantiles
        tau_embeddings = self.cosine_net(taus)
        quantiles = self.quantile_net(embeddings, tau_embeddings)  # (batch_size, K, num_actions)

        # Compute expectations of value distributions.
        return torch.mean(quantiles, dim=1)  # (batch_size, num_actions)


env_id = "PongNoFrameskip-v4"

total_timesteps = 7_000  # 50_000_000
learning_starts = 3_000  # 50_000

start_e = 1
end_e = 0.01
exploration_fraction = 25
slope = (end_e - start_e) / (exploration_fraction * total_timesteps)

train_frequency = 4
batch_size = 32
gamma = 0.99
learning_rate = 5e-5
target_network_frequency = 10_000

N = 64
N_dash = 64
K = 32
num_cosines = 64
kappa = 1.0
memory_size = 1_000_000

# Env setup
env = utils.AtariWrapper(gym.make(env_id))
env = gym.wrappers.RecordEpisodeStatistics(env)
env = TorchWrapper(env)

# Seeding
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
env.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network setup
online_net = IQN(env, K=K, num_cosines=num_cosines).to(device)
optimizer = optim.Adam(online_net.parameters(), lr=learning_rate, eps=1e-2 / batch_size)
target_net = IQN(env, K=K, num_cosines=num_cosines).to(device)
target_net.load_state_dict(online_net.state_dict())

# Storage setup
observations = torch.empty((memory_size, *env.observation_space.shape), dtype=torch.uint8)
actions = torch.empty(memory_size, dtype=torch.long)
rewards = torch.empty(memory_size, dtype=torch.float32)
terminated = torch.empty(memory_size, dtype=torch.bool)

# Initiate the envrionment and store the inital observation
observation = env.reset()
global_step = 0
observations[global_step % memory_size] = observation

# Loop
while global_step < total_timesteps:
    # Update exploration rate
    epsilon = max(slope * global_step + start_e, end_e)

    if global_step < learning_starts or np.random.rand() < epsilon:
        action = torch.tensor(env.action_space.sample())
    else:
        observation_ = observation.unsqueeze(0).to(device).float().permute(0, 3, 1, 2) / 255.0
        with torch.no_grad():
            embeddings = online_net.compute_embeddings(observation_)
            action = online_net.compute_q_values(embeddings).argmax()

    # Store
    actions[global_step % memory_size] = action

    # Step
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()

    # Update count
    global_step += 1

    observations[global_step % memory_size] = observation
    rewards[global_step % memory_size] = reward
    terminated[global_step % memory_size] = done and not info.get("TimeLimit.truncated", False)

    if "episode" in info.keys():
        print(f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}")

    # Optimize the agent
    if global_step >= learning_starts:
        if global_step % train_frequency == 0:
            upper = min(global_step, memory_size)
            batch_inds = np.random.randint(global_step, size=batch_size)

            b_observations = observations[batch_inds].to(device)
            b_actions = actions[batch_inds].to(device)
            b_next_observations = observations[(batch_inds + 1) % memory_size].to(device)
            b_rewards = rewards[(batch_inds + 1) % memory_size].to(device)
            b_terminated = terminated[(batch_inds + 1) % memory_size].to(device)

            # Normalize images and make them channel-first
            b_observations = b_observations.float().permute(0, 3, 1, 2) / 255.0
            b_next_observations = b_next_observations.float().permute(0, 3, 1, 2) / 255.0

            # Compute the embeddings
            embeddings = online_net.compute_embeddings(b_observations)

            # Sample fractions
            taus = torch.rand(batch_size, N, dtype=embeddings.dtype, device=device)

            # Compute quantile values of current observations and actions at tau_hats
            current_sa_quantiles = online_net.compute_quantile_at_action(embeddings, taus, b_actions)

            # Compute Q values of next observations
            next_embeddings = target_net.compute_embeddings(b_next_observations)
            next_q_value = target_net.compute_q_values(next_embeddings)

            # Compute greedy actions
            next_actions = torch.argmax(next_q_value, dim=1)

            # Sample next fractions
            tau_dashes = torch.rand(batch_size, N_dash, dtype=embeddings.dtype, device=device)

            # Compute quantile values of next observations and next actions
            next_sa_quantiles = online_net.compute_quantile_at_action(next_embeddings, tau_dashes, next_actions)

            # Compute target quantile values (batch_size, 1, N_dash)
            target_sa_quantiles = b_rewards[..., None] + torch.logical_not(b_terminated)[..., None] * gamma * next_sa_quantiles

            # TD-error is the cross differnce between the target quantiles and the currents quantiles
            td_errors = target_sa_quantiles.unsqueeze(-2).detach() - current_sa_quantiles.unsqueeze(-1)

            # Compute quantile Huber loss
            huber_loss = torch.where(td_errors.abs() <= kappa, 0.5 * td_errors.pow(2), kappa * (td_errors.abs() - 0.5 * kappa))
            quantile_huber_loss = torch.abs(taus[..., None] - (td_errors < 0).float()) * huber_loss
            quantile_loss = torch.mean(quantile_huber_loss)

            optimizer.zero_grad()
            quantile_loss.backward()
            optimizer.step()

        # Update the target network
        if global_step % target_network_frequency == 0:
            target_net.load_state_dict(online_net.state_dict())

env.close()
