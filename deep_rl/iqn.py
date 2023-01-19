import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam

import utils


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class DQNBase(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        ).apply(initialize_weights_he)

    def forward(self, observations: Tensor) -> Tensor:
        return self.net(observations)


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
        self.dqn_net = DQNBase(num_channels=num_channels)
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


def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(td_errors.abs() <= kappa, 0.5 * td_errors.pow(2), kappa * (td_errors.abs() - 0.5 * kappa))


def calculate_quantile_huber_loss(td_errors, taus, kappa=1.0):
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
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


class IQNAgent:
    def __init__(
        self,
        env,
        num_steps=50_000_000,
        batch_size=32,
        N=64,
        N_dash=64,
        K=32,
        num_cosines=64,
        kappa=1.0,
        lr=5e-5,
        memory_size=1_000_000,
        gamma=0.99,
        update_interval=4,
        target_update_interval=10_000,
        start_steps=50_000,
        epsilon_train=0.01,
        epsilon_decay_steps=250_000,
        log_interval=100,
        max_episode_steps=27_000,
        cuda=True,
        seed=0,
    ):
        self.env = env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)

        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

        # Replay memory which is memory-efficient to store stacked frames.
        self.memory = utils.LazyMultiStepMemory(memory_size, self.env.observation_space.shape, self.device, gamma)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_actions = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.log_interval = log_interval
        self.gamma = gamma
        self.start_steps = start_steps
        self.epsilon_train = epsilon_train
        self.epsilon_decay_steps = epsilon_decay_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.max_episode_steps = max_episode_steps

        # Online network.
        self.online_net = IQN(
            num_channels=env.observation_space.shape[2], num_actions=self.num_actions, K=K, num_cosines=num_cosines
        ).to(self.device)
        # Target network.
        self.target_net = IQN(
            num_channels=env.observation_space.shape[2], num_actions=self.num_actions, K=K, num_cosines=num_cosines
        ).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.target_net.load_state_dict(self.online_net.state_dict())
        # Disable calculations of gradients of the target network.
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optim = Adam(self.online_net.parameters(), lr=lr, eps=1e-2 / batch_size)

        self.N = N
        self.N_dash = N_dash
        self.K = K
        self.num_cosines = num_cosines
        self.kappa = kappa

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def train_episode(self):
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.0
        episode_steps = 0

        done = False
        observation = self.env.reset()

        while (not done) and episode_steps <= self.max_episode_steps:
            # Use e-greedy for evaluation.
            epsilon = max(self.epsilon_train, 1 - (1 - self.epsilon_train) / self.epsilon_decay_steps * self.steps)
            if self.steps < self.start_steps or np.random.rand() < epsilon:
                action = self.env.action_space.sample()
            else:
                observation_ = torch.ByteTensor(observation).unsqueeze(0).to(self.device).float().permute(0, 3, 1, 2) / 255.0
                with torch.no_grad():
                    embeddings = self.online_net.compute_embeddings(observation_)
                    action = self.online_net.compute_q_values(embeddings).argmax().item()

            next_observation, reward, done, _ = self.env.step(action)

            # To calculate efficiently, I just set priority=max_priority here.
            self.memory.append(observation, action, reward, next_observation, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            observation = next_observation

            if self.steps % self.target_update_interval == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            if self.steps % self.update_interval == 0 and self.steps >= self.start_steps:
                self.learn()

        print(f"Episode: {self.episodes:<4}  " f"episode steps: {episode_steps:<4}  " f"return: {episode_return:<5.1f}")

    def __del__(self):
        self.env.close()

    def learn(self):
        self.learning_steps += 1
        observations, actions, rewards, next_observations, dones = self.memory.sample(self.batch_size)

        # Compute the embeddings
        embeddings = self.online_net.compute_embeddings(observations)

        # Sample fractions.
        taus = torch.rand(self.batch_size, self.N, dtype=embeddings.dtype, device=embeddings.device)
        # Calculate quantile values of current states and actions at tau_hats.
        current_sa_quantiles = evaluate_quantile_at_action(
            self.online_net.compute_quantiles(embeddings, taus), actions
        )  # (self.batch_size, self.N, 1)

        with torch.no_grad():
            # Calculate Q values of next states.
            next_embeddings = self.target_net.compute_embeddings(next_observations)
            next_q_value = self.target_net.compute_q_values(next_embeddings)

            # Calculate greedy actions.
            next_actions = torch.argmax(next_q_value, dim=1, keepdim=True)  # (self.batch_size, 1)

            # Sample next fractions.
            tau_dashes = torch.rand(self.batch_size, self.N_dash, dtype=embeddings.dtype, device=embeddings.device)

            # Calculate quantile values of next states and next actions.
            next_sa_quantiles = evaluate_quantile_at_action(
                self.target_net.compute_quantiles(next_embeddings, tau_dashes), next_actions
            ).transpose(
                1, 2
            )  # (self.batch_size, 1, self.N_dash)

            # Calculate target quantile values.
            target_sa_quantiles = (
                rewards[..., None] + (1.0 - dones[..., None]) * self.gamma * next_sa_quantiles
            )  # (self.batch_size, 1, self.N_dash)

        td_errors = target_sa_quantiles - current_sa_quantiles
        quantile_loss = calculate_quantile_huber_loss(td_errors, taus, self.kappa)

        self.optim.zero_grad()
        quantile_loss.backward()
        self.optim.step()


if __name__ == "__main__":
    import gym

    net = CosineEmbeddingNetwork()
    x = torch.randint(0, 3, (4, 3))
    print(net(x).shape)
    env_id = "PongNoFrameskip-v4"

    # Create environments.
    env = utils.AtariWrapper(gym.make(env_id))

    # Create the agent and run.
    agent = IQNAgent(env=env, num_steps=10_000, start_steps=5_000)
    agent.run()
