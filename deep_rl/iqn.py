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


class FeaturesExtractor(nn.Module):
    def __init__(self, env: gym.Env) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(env.observation_space.shape[2], 32, kernel_size=8, stride=4),  # (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # (64 * 7 * 7,)
        ).apply(initialize_weights_he)

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input)


class CosineEmbeddingNetwork(nn.Module):
    """
    Computes the embeddings of tau values using cosine functions.

    Take a tensor of shape (batch_size, N) representing the tau values, and return
    a tensor of shape (batch_size, N, embedding_dim) representing the embeddings of tau values.

    Args:
        num_cosines (int): Number of cosines to use for embedding
        embedding_dim (int): Dimension of the embedding
    """

    def __init__(self, num_cosines: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_cosines, embedding_dim),
            nn.ReLU(),
        )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus: Tensor) -> Tensor:
        # Compute cos(i * pi * tau)
        i_pi = np.pi * torch.arange(start=1, end=self.num_cosines + 1, device=taus.device)
        i_pi = i_pi.reshape(1, 1, self.num_cosines)  # (1, 1, num_cosines)
        taus = torch.unsqueeze(taus, dim=-1)  # (batch_size, N, 1)
        cosines = torch.cos(taus * i_pi)  # (batch_size, N, num_cosines)

        # Compute embeddings of taus
        cosines = torch.flatten(cosines, end_dim=1)  # (batch_size * N, num_cosines)
        tau_embeddings = self.net(cosines)  # (batch_size * N, embedding_dim)
        return torch.unflatten(tau_embeddings, dim=0, sizes=(-1, taus.shape[1]))  # (batch_size, N, embedding_dim)


class QuantileNetwork(nn.Module):
    """
    Compute the quantile values of actions given the embeddings and tau values.

    Take as input the embedding tensor of shape (batch_size, embedding_dim) and the
    tau embeddings tensor of shape (batch_size, M, embedding_dim) and returns a tensor
    of shape (batch_size, M, num_actions) quantile values of actions.

    Args:
        num_actions (int): Number of actions
        embedding_dim (int): Dimension of the embeddings
    """

    def __init__(self, num_actions: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, embeddings: Tensor, tau_embeddings: Tensor) -> Tensor:
        # Compute the embeddings and taus
        embeddings = torch.unsqueeze(embeddings, dim=1)  # (batch_size, 1, embedding_dim)
        embeddings = embeddings * tau_embeddings  # (batch_size, M, embedding_dim)

        # Compute the quantile values
        embeddings = torch.flatten(embeddings, end_dim=1)  # (batch_size * M, embedding_dim)
        quantiles = self.net(embeddings)
        return torch.unflatten(quantiles, dim=0, sizes=(-1, tau_embeddings.shape[1]))  # (batch_size, M, num_actions)


env_id = "PongNoFrameskip-v4"

total_timesteps = 50_000_000
learning_starts = 50_000

start_e = 1
end_e = 0.01
exploration_fraction = 0.005
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
embedding_dim = 7 * 7 * 64
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
online_features_extractor = FeaturesExtractor(env).to(device)
online_cosine_net = CosineEmbeddingNetwork(num_cosines=num_cosines, embedding_dim=embedding_dim).to(device)
online_quantile_net = QuantileNetwork(num_actions=env.action_space.n, embedding_dim=embedding_dim).to(device)

target_features_extractor = FeaturesExtractor(env).to(device)
target_cosine_net = CosineEmbeddingNetwork(num_cosines=num_cosines, embedding_dim=embedding_dim).to(device)
target_quantile_net = QuantileNetwork(num_actions=env.action_space.n, embedding_dim=embedding_dim).to(device)

# Initialize the weights
target_features_extractor.load_state_dict(online_features_extractor.state_dict())
target_cosine_net.load_state_dict(online_cosine_net.state_dict())
target_quantile_net.load_state_dict(online_quantile_net.state_dict())

# Instanciate the optimizer
parameters = [*online_features_extractor.parameters(), *online_cosine_net.parameters(), *online_quantile_net.parameters()]
optimizer = optim.Adam(parameters, lr=learning_rate, eps=1e-2 / batch_size)

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
        # Normalize image and make it channel-first
        observation_ = observation.to(device).unsqueeze(0).float().permute(0, 3, 1, 2) / 255.0
        with torch.no_grad():
            # Compute the embedding, sample fractions and compute quantiles
            embeddings = online_features_extractor(observation_)
            taus = torch.rand(embeddings.shape[0], K, device=embeddings.device)
            tau_embeddings = online_cosine_net(taus)
            quantiles = online_quantile_net(embeddings, tau_embeddings)  # (batch_size, K, num_actions)
        q_values = torch.mean(quantiles, dim=1)  # (batch_size, num_actions)
        action = torch.argmax(q_values)

    # Store
    actions[global_step % memory_size] = action

    # Step
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()

    # Update count
    global_step += 1

    # Store
    observations[global_step % memory_size] = observation
    rewards[global_step % memory_size] = reward
    terminated[global_step % memory_size] = done and not info.get("TimeLimit.truncated", False)

    if "episode" in info.keys():
        print(f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}")

    # Optimize the agent
    if global_step >= learning_starts:
        if global_step % train_frequency == 0:
            upper = min(global_step, memory_size)
            batch_inds = np.random.randint(upper, size=batch_size)

            b_observations = observations[batch_inds].to(device)
            b_actions = actions[batch_inds].to(device)
            b_next_observations = observations[(batch_inds + 1) % memory_size].to(device)
            b_rewards = rewards[(batch_inds + 1) % memory_size].to(device)
            b_terminated = terminated[(batch_inds + 1) % memory_size].to(device)

            # Normalize images and make them channel-first (batch_size, H, W, 1) to (batch_size, 1, H, W)
            b_observations = b_observations.float().permute(0, 3, 1, 2) / 255.0
            b_next_observations = b_next_observations.float().permute(0, 3, 1, 2) / 255.0

            # Sample fractions and compute quantile values of current observations and actions at taus
            embeddings = online_features_extractor(b_observations)
            taus = torch.rand(batch_size, N, device=device)
            tau_embeddings = online_cosine_net(taus)
            quantiles = online_quantile_net(embeddings, tau_embeddings)

            # Compute quantile values at specified actions. The notation seems eavy notation,
            # but just select value of s_quantile (B, N, num_quantiles) with action_indexes (batch_size, K).
            # Output shape is thus (batch_size, K)
            action_index = b_actions[..., None].expand(-1, quantiles.shape[1])  # Copy actions to a shape (batch_size, N)
            current_action_quantiles = quantiles.gather(dim=2, index=action_index.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                # Compute Q values of next observations
                next_embeddings = target_features_extractor(b_next_observations)
                next_taus = torch.rand(batch_size, K, device=device)
                next_tau_embeddings = target_cosine_net(next_taus)
                next_quantiles = target_quantile_net(next_embeddings, next_tau_embeddings)  # (batch_size, K, num_actions)

                # Compute greedy actions
                next_q_values = torch.mean(next_quantiles, dim=1)  # (batch_size, num_actions)
                next_actions = torch.argmax(next_q_values, dim=1)

                # Compute next quantiles
                tau_dashes = torch.rand(batch_size, N_dash, device=device)
                tau_dashes_embeddings = target_cosine_net(tau_dashes)
                next_quantiles = target_quantile_net(next_embeddings, tau_dashes_embeddings)

                # Compute quantile values at specified actions. The notation seems eavy notation,
                # but just select value of s_quantile (B, N, num_quantiles) with action_indexes (batch_size, K).
                # Output shape is thus (batch_size, K)
                next_action_index = next_actions[..., None].expand(-1, next_quantiles.shape[1])
                next_action_quantiles = next_quantiles.gather(dim=2, index=next_action_index.unsqueeze(-1)).squeeze(-1)

                # Compute target quantile values (batch_size, N_dash)
                target_action_quantiles = (
                    b_rewards[..., None] + torch.logical_not(b_terminated)[..., None] * gamma * next_action_quantiles
                )

            # TD-error is the cross differnce between the target quantiles and the currents quantiles
            td_errors = target_action_quantiles.unsqueeze(-2) - current_action_quantiles.unsqueeze(-1)

            # Compute quantile Huber loss
            huber_loss = torch.where(td_errors.abs() <= kappa, 0.5 * td_errors.pow(2), kappa * (td_errors.abs() - 0.5 * kappa))
            quantile_huber_loss = torch.abs(taus[..., None] - (td_errors < 0).float()) * huber_loss
            quantile_loss = torch.mean(quantile_huber_loss)

            optimizer.zero_grad()
            quantile_loss.backward()
            optimizer.step()

        # Update the target network
        if global_step % target_network_frequency == 0:
            target_features_extractor.load_state_dict(online_features_extractor.state_dict())
            target_cosine_net.load_state_dict(online_cosine_net.state_dict())
            target_quantile_net.load_state_dict(online_quantile_net.state_dict())

env.close()
