import time
from typing import Any, Dict, Tuple

import envpool
import gym
import numpy as np
import torch
from envpool.python.protocol import EnvPool
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset


def _dict_to_tensor(value):
    if isinstance(value, dict):
        return {key: _dict_to_tensor(value) for key, value in value.items()}
    elif isinstance(value, np.ndarray):
        return torch.from_numpy(value)


class TorchWrapper:
    """
    Torch wrapper. Actions and observations are Tensors instead of arrays.
    """

    def __init__(self, env: EnvPool) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        action = action.cpu().numpy()
        observation, reward, done, info = self.env.step(action)
        return (torch.from_numpy(observation), torch.from_numpy(reward), torch.from_numpy(done), _dict_to_tensor(info))

    def reset(self) -> Tensor:
        observation = self.env.reset()
        return torch.from_numpy(observation)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Buffer(Dataset):
    def __init__(self, memory_size, num_envs, observation_shape):
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.observations = torch.empty((memory_size, num_envs, *observation_shape), dtype=torch.uint8)
        self.actions = torch.empty((memory_size, num_envs), dtype=torch.long)
        self.rewards = torch.empty((memory_size, num_envs), dtype=torch.float32)
        self.terminated = torch.empty((memory_size, num_envs), dtype=torch.bool)
        self.pos = 0

    def __len__(self):
        return self.pos

    def __getitem__(self, idx):
        batch_inds = idx / self.num_envs
        env_inds = idx % self.num_envs
        b_observations = self.observations[batch_inds, env_inds]
        b_actions = self.actions[batch_inds, env_inds]
        b_next_observations = self.observations[(batch_inds + 1) % self.memory_size, env_inds]
        b_rewards = self.rewards[(batch_inds + 1) % self.memory_size, env_inds]
        b_terminated = self.terminated[(batch_inds + 1) % self.memory_size, env_inds]

        return (
            b_observations,
            b_actions,
            b_next_observations,
            b_rewards,
            b_terminated,
        )


class FeaturesExtractor(nn.Module):
    def __init__(self, env: gym.Env) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(env.observation_space.shape[0], 32, kernel_size=8, stride=4),  # (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # (64 * 7 * 7,)
        ).apply(initialize_weights_he)

    def forward(self, input: Tensor) -> Tensor:
        return self.net(input / 255.0)


class CosineEmbeddingNetwork(nn.Module):
    """
    Computes the embeddings of tau values using cosine functions.

    Take a tensor of shape (batch_size, num_tau_samples) representing the tau values, and return
    a tensor of shape (batch_size, num_tau_samples, embedding_dim) representing the embeddings of tau values.

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
        i_pi = np.pi * torch.arange(start=1, end=num_cosines + 1).reshape(1, 1, num_cosines)  # (1, 1, num_cosines)
        self.register_buffer("i_pi", i_pi)

    def forward(self, taus: Tensor) -> Tensor:
        # Compute cos(i * pi * tau)
        taus = torch.unsqueeze(taus, dim=-1)  # (batch_size, num_tau_samples, 1)
        cosines = torch.cos(taus * self.i_pi)  # (batch_size, num_tau_samples, num_cosines)

        # Compute embeddings of taus
        cosines = torch.flatten(cosines, end_dim=1)  # (batch_size * num_tau_samples, num_cosines)
        tau_embeddings = self.net(cosines)  # (batch_size * num_tau_samples, embedding_dim)
        return torch.unflatten(
            tau_embeddings, dim=0, sizes=(-1, taus.shape[1])
        )  # (batch_size, num_tau_samples, embedding_dim)


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

if __name__ == "__main__":
    env_id = "Pong-v5"

    total_timesteps = 85_000  # 10_000_000
    num_envs = 32
    learning_starts = 50_000 // num_envs


    final_epsilon = 0.01
    epsilon_decay_steps = 250_000 // num_envs
    slope = -(1.0 - final_epsilon) / epsilon_decay_steps

    train_frequency = 1
    grad_steps = num_envs // 4
    batch_size = 32
    gamma = 0.99
    learning_rate = 5e-5
    target_network_frequency = 10_000 // num_envs

    num_tau_samples = 64
    num_tau_prime_samples = 64
    num_quantile_samples = 32
    num_cosines = 64
    embedding_dim = 7 * 7 * 64
    kappa = 1.0
    memory_size = 1_000_000 // num_envs

    seed = 0

    # Env setup
    env = envpool.make(env_id, env_type="gym", num_envs=num_envs, episodic_life=True, reward_clip=True, seed=seed)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env = TorchWrapper(env)

    # Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    # observations = torch.empty((memory_size, num_envs, *env.observation_space.shape), dtype=torch.uint8)
    # actions = torch.empty((memory_size, num_envs), dtype=torch.long)
    # rewards = torch.empty((memory_size, num_envs), dtype=torch.float32)
    # terminated = torch.empty((memory_size, num_envs), dtype=torch.bool)
    buffer = Buffer(memory_size, num_envs, env.observation_space.shape)
    loader = None

    # Initiate the envrionment and store the inital observation
    observation = env.reset()
    global_step = 0
    buffer.observations[global_step % memory_size] = observation

    start_time = time.time()

    # Loop
    while global_step * num_envs < total_timesteps:
        # Update exploration rate
        epsilon = max(1.0 + slope * global_step, final_epsilon)

        action = torch.tensor([env.action_space.sample() for _ in range(num_envs)], device=device)
        if global_step > learning_starts:
            with torch.no_grad():
                embeddings = online_features_extractor(observation.to(device, non_blocking=True))  # (num_envs, embedding_dim)
                taus = torch.rand(num_envs, num_quantile_samples, device=device)  # (num_envs, num_quantile_samples)
                tau_embeddings = online_cosine_net(taus)  # (num_envs, num_quantile_samples, embedding_dim)
                quantiles = online_quantile_net(embeddings, tau_embeddings)  # (num_envs, num_quantile_samples, num_actions)
                q_values = torch.mean(quantiles, dim=1)  # (num_envs, num_actions)
                pred_action = torch.argmax(q_values, dim=1)  # (num_envs,)
                random_action_idxs = torch.rand(num_envs, device=device) < epsilon
                action = torch.where(random_action_idxs, action, pred_action)

        # Store
        buffer.actions[global_step % memory_size] = action

        # Step
        observation, reward, done, info = env.step(action)

        # Update count
        global_step += 1
        buffer.pos = global_step % memory_size

        # Store
        buffer.observations[global_step % memory_size] = observation
        buffer.rewards[global_step % memory_size] = reward
        buffer.terminated[global_step % memory_size] = torch.logical_and(done, torch.logical_not(info.get("TimeLimit.truncated")))

        for env_idx in range(num_envs):
            if done[env_idx]:
                elapsed_step = info["elapsed_step"][env_idx]
                episode_steps = torch.arange(global_step - elapsed_step + 1, global_step) % memory_size
                cumulative_reward = torch.sum(buffer.rewards[episode_steps, env_idx])
                print(
                    f"global_step={global_step*num_envs}, episodic_return={cumulative_reward:.2f}, fps={global_step * num_envs / (time.time() - start_time):.2f}"
                )

        # Optimize the agent
        if global_step >= learning_starts:
            if loader is None:
                loader = DataLoader(buffer, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            if global_step % train_frequency == 0:
                for _ in range(grad_steps):
                    upper = min(global_step, memory_size)
                    batch_inds = np.random.randint(upper, size=batch_size)
                    env_inds = np.random.randint(num_envs, size=batch_size)

                    # b_observations = observations[batch_inds, env_inds].to(device, non_blocking=True)
                    # b_actions = actions[batch_inds, env_inds].to(device, non_blocking=True)
                    # b_next_observations = observations[(batch_inds + 1) % memory_size, env_inds].to(device, non_blocking=True)
                    # b_rewards = rewards[(batch_inds + 1) % memory_size, env_inds].to(device, non_blocking=True)
                    # b_terminated = terminated[(batch_inds + 1) % memory_size, env_inds].to(device, non_blocking=True)
                    b_observations, b_actions, b_next_observations, b_rewards, b_terminated = next(iter(loader))

                    # Sample fractions and compute quantile values of current observations and actions at taus
                    embeddings = online_features_extractor(b_observations)
                    taus = torch.rand(batch_size, num_tau_samples, device=device)
                    tau_embeddings = online_cosine_net(taus)
                    quantiles = online_quantile_net(embeddings, tau_embeddings)

                    # Compute quantile values at specified actions. The notation seems eavy notation,
                    # but just select value of s_quantile (batch_size, num_tau_samples, num_quantiles) with
                    # action_indexes (batch_size, num_quantile_samples).
                    # Output shape is thus (batch_size, num_quantile_samples)
                    action_index = b_actions[..., None].expand(-1, num_tau_samples)  # Expand to (batch_size, num_tau_samples)
                    current_action_quantiles = quantiles.gather(dim=2, index=action_index.unsqueeze(-1)).squeeze(-1)

                    # Compute Q values of next observations
                    next_embeddings = target_features_extractor(b_next_observations)
                    next_taus = torch.rand(batch_size, num_quantile_samples, device=device)
                    next_tau_embeddings = target_cosine_net(next_taus)
                    next_quantiles = target_quantile_net(
                        next_embeddings, next_tau_embeddings
                    )  # (batch_size, num_quantile_samples, num_actions)

                    # Compute greedy actions
                    next_q_values = torch.mean(next_quantiles, dim=1)  # (batch_size, num_actions)
                    next_actions = torch.argmax(next_q_values, dim=1)  # (batch_size,)

                    # Compute next quantiles
                    tau_dashes = torch.rand(batch_size, num_tau_prime_samples, device=device)
                    tau_dashes_embeddings = target_cosine_net(tau_dashes)
                    next_quantiles = target_quantile_net(next_embeddings, tau_dashes_embeddings)

                    # Compute quantile values at specified actions. The notation seems eavy notation,
                    # but just select value of s_quantile (batch_size, num_tau_samples, num_quantiles)
                    # with action_indexes (batch_size, num_quantile_samples).
                    # Output shape is thus (batch_size, num_quantile_samples).
                    next_action_index = next_actions[..., None].expand(-1, num_tau_prime_samples)
                    next_action_quantiles = next_quantiles.gather(dim=2, index=next_action_index.unsqueeze(-1)).squeeze(-1)

                    # Compute target quantile values (batch_size, num_tau_prime_samples)
                    target_action_quantiles = (
                        b_rewards[..., None] + torch.logical_not(b_terminated)[..., None] * gamma * next_action_quantiles
                    )

                    # TD-error is the cross differnce between the target quantiles and the currents quantiles
                    td_errors = target_action_quantiles.unsqueeze(-2).detach() - current_action_quantiles.unsqueeze(-1)

                    # Compute quantile Huber loss
                    huber_loss = torch.where(
                        torch.abs(td_errors) <= kappa, td_errors**2, kappa * (torch.abs(td_errors) - 0.5 * kappa)
                    )
                    quantile_huber_loss = torch.abs(taus[..., None] - (td_errors.detach() < 0).float()) * huber_loss / kappa
                    batch_quantile_huber_loss = torch.sum(quantile_huber_loss, dim=1)
                    quantile_loss = torch.mean(batch_quantile_huber_loss)

                    optimizer.zero_grad()
                    quantile_loss.backward()
                    optimizer.step()

            # Update the target network
            if global_step % target_network_frequency == 0:
                target_features_extractor.load_state_dict(online_features_extractor.state_dict())
                target_cosine_net.load_state_dict(online_cosine_net.state_dict())
                target_quantile_net.load_state_dict(online_quantile_net.state_dict())

    env.close()
