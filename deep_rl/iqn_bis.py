from collections import deque

import cv2
import gym
import numpy as np
import torch
from gym import spaces
from torch import nn, optim
from utils import NoopResetWrapper, MaxAndSkipWrapper, EpisodicLifeWrapper, FireResetWrapper, ClipRewardWrapper
import wandb

cv2.ocl.setUseOpenCL(False)


class WarpFramePyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=env.observation_space.dtype
        )

    def observation(self, frame):
        """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStackPyTorch(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape

        self.observation_space = spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(shp[0], shp[1], shp[2] * n_frames),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return np.concatenate(np.array(self.frames), axis=-1)


class AtariWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        env = NoopResetWrapper(env, noop_max=30)
        env = MaxAndSkipWrapper(env, frame_skip=4)
        env = EpisodicLifeWrapper(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetWrapper(env)
        env = WarpFramePyTorch(env)
        env = ClipRewardWrapper(env)
        env = FrameStackPyTorch(env, 4)
        super().__init__(env)


class LazyMultiStepMemory:
    def __init__(self, capacity, observation_shape, device):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.device = device
        self.observation = []
        self.next_observation = []
        self.action = np.empty((self.capacity, 1), dtype=np.int64)
        self.reward = np.empty((self.capacity, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, 1), dtype=np.float32)
        self._n = 0
        self._p = 0

    def append(self, observation, action, reward, next_observation, done):
        self.observation.append(observation)
        self.next_observation.append(next_observation)
        self.action[self._p] = action
        self.reward[self._p] = reward
        self.done[self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        while len(self.observation) > self.capacity:
            del self.observation[0]
            del self.next_observation[0]

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self.observation), size=batch_size)
        bias = -self._p if self._n == self.capacity else 0

        observations = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)
        next_observations = np.empty((batch_size, *self.observation_shape), dtype=np.uint8)

        for i, index in enumerate(indices):
            _index = np.mod(index + bias, self.capacity)
            observations[i, ...] = self.observation[_index]
            next_observations[i, ...] = self.next_observation[_index]

        observations = torch.ByteTensor(observations).to(self.device).transpose(3, 1).float() / 255.0
        next_observations = torch.ByteTensor(next_observations).to(self.device).transpose(3, 1).float() / 255.0
        actions = torch.LongTensor(self.action[indices]).to(self.device)
        rewards = torch.FloatTensor(self.reward[indices]).to(self.device)
        dones = torch.FloatTensor(self.done[indices]).to(self.device)

        return observations, actions, rewards, next_observations, dones


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class DQNBase(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        ).apply(initialize_weights_he)

    def forward(self, observations):
        return self.net(observations)


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, num_cosines=64, embedding_dim=7 * 7 * 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_cosines, embedding_dim), nn.ReLU())
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(start=1, end=self.num_cosines + 1, dtype=taus.dtype, device=taus.device).view(
            1, 1, self.num_cosines
        )

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(taus.view(batch_size, N, 1) * i_pi).view(batch_size * N, self.num_cosines)

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(batch_size, N, self.embedding_dim)

        return tau_embeddings


class QuantileNetwork(nn.Module):
    def __init__(self, num_actions, embedding_dim=7 * 7 * 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

    def forward(self, observation_embeddings, tau_embeddings):

        # NOTE: Because variable taus correspond to either \tau or \hat \tau
        # in the paper, N isn't neccesarily the same as fqf.N.
        batch_size = observation_embeddings.shape[0]
        N = tau_embeddings.shape[1]

        # Reshape into (batch_size, 1, embedding_dim).
        observation_embeddings = observation_embeddings.view(batch_size, 1, self.embedding_dim)

        # Calculate embeddings of observations and taus.
        embeddings = (observation_embeddings * tau_embeddings).view(batch_size * N, self.embedding_dim)

        # Calculate quantile values.

        quantiles = self.net(embeddings)
        return quantiles.view(batch_size, N, self.num_actions)


class IQN(nn.Module):
    def __init__(self, num_channels, num_actions, K=32, num_cosines=32, embedding_dim=7 * 7 * 64):
        super().__init__()
        self.dqn_net = DQNBase(num_channels=num_channels)
        self.cosine_net = CosineEmbeddingNetwork(num_cosines=num_cosines, embedding_dim=embedding_dim)
        self.quantile_net = QuantileNetwork(num_actions=num_actions)
        self.K = K

    def calculate_observation_embeddings(self, observations):
        return self.dqn_net(observations)

    def calculate_quantiles(self, taus, observations=None, observation_embeddings=None):
        if observation_embeddings is None:
            observation_embeddings = self.dqn_net(observations)

        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(observation_embeddings, tau_embeddings)

    def calculate_q(self, observations=None, observation_embeddings=None):
        batch_size = observations.shape[0] if observations is not None else observation_embeddings.shape[0]
        if observation_embeddings is None:
            observation_embeddings = self.dqn_net(observations)
        taus = torch.rand(batch_size, self.K, dtype=observation_embeddings.dtype, device=observation_embeddings.device)
        quantiles = self.calculate_quantiles(taus, observation_embeddings=observation_embeddings)
        return quantiles.mean(dim=1)


def evaluate_quantile_at_action(s_quantiles, actions):
    batch_size = s_quantiles.shape[0]
    N = s_quantiles.shape[1]
    action_index = actions[..., None].expand(batch_size, N, 1)
    return s_quantiles.gather(dim=2, index=action_index)


# Create environments.
env_id = "PongNoFrameskip-v4"

env = gym.make(env_id)
env = AtariWrapper(env)

# Create the agent and run.

batch_size = 32
N = 64
N_dash = 64
K = 32
num_cosines = 64
kappa = 1.0
memory_size = 1_000_000
gamma = 0.99
update_interval = 4
epsilon_train = 0.01
epsilon_eval = 0.001
max_episode_steps = 27_000
seed = 0

# num_steps = 50_000  # 50_000_000
# lr = 5e-4  # 5e-5
# target_update_interval = 1_000  # 10_000
# start_steps = 500  # 50_000
# epsilon_decay_steps = 2_000  # 250_000
# eval_interval = 2_000  # 250_000
# num_eval_steps = 2_000  # 125_000

num_steps = 10_000_000
lr = 5e-5
target_update_interval = 10_000
start_steps = 50_000
epsilon_decay_steps = 250_000
eval_interval = 250_000
num_eval_steps = 125_000

wandb.init(project="IQN")

torch.manual_seed(seed)
np.random.seed(seed)
env.seed(seed)
env.observation_space.seed(seed)
env.action_space.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay memory which is memory-efficient to store stacked frames.
memory = LazyMultiStepMemory(memory_size, env.observation_space.shape, device)
global_step = 0
episodes = 0
num_actions = env.action_space.n

slope = (epsilon_train - 1.0) / epsilon_decay_steps

# Online network.
online_net = IQN(num_channels=env.observation_space.shape[2], num_actions=num_actions, K=K, num_cosines=num_cosines).to(device)
# Target network.
target_net = IQN(num_channels=env.observation_space.shape[2], num_actions=num_actions, K=K, num_cosines=num_cosines).to(device)

# Copy parameters of the learning network to the target network.
target_net.load_state_dict(online_net.state_dict())

optimizer = optim.Adam(online_net.parameters(), lr=lr, eps=1e-2 / batch_size)


while True:
    episodes += 1
    episode_return = 0.0
    episode_steps = 0

    done = False
    observation = env.reset()

    while (not done) and episode_steps <= max_episode_steps:
        if global_step < start_steps or np.random.rand() < slope * min(global_step, epsilon_decay_steps) + 1.0:
            action = env.action_space.sample()
        else:
            observation_ = torch.ByteTensor(np.array(observation)).unsqueeze(0).transpose(3, 1).to(device).float() / 255.0
            with torch.no_grad():
                action = online_net.calculate_q(observations=observation_).argmax().item()

        next_observation, reward, done, _ = env.step(action)
        memory.append(observation, action, reward, next_observation, done)

        global_step += 1
        episode_steps += 1
        episode_return += reward
        observation = next_observation

        if global_step % target_update_interval == 0:
            target_net.load_state_dict(online_net.state_dict())

        if global_step % update_interval == 0 and global_step >= start_steps:
            b_observations, b_actions, b_rewards, b_next_observations, b_dones = memory.sample(batch_size)

            # Calculate features of observations.
            observation_embeddings = online_net.calculate_observation_embeddings(b_observations)

            # Sample fractions.
            taus = torch.rand(batch_size, N, dtype=observation_embeddings.dtype, device=observation_embeddings.device)

            # Calculate quantile values of current observations and actions at tau_hats.
            current_sa_quantiles = evaluate_quantile_at_action(
                online_net.calculate_quantiles(taus, observation_embeddings=observation_embeddings), b_actions
            )

            with torch.no_grad():
                # Calculate Q values of next observations.
                next_observation_embeddings = target_net.calculate_observation_embeddings(b_next_observations)
                next_q = target_net.calculate_q(observation_embeddings=next_observation_embeddings)

                # Calculate greedy actions.
                next_actions = torch.argmax(next_q, dim=1, keepdim=True)

                # Sample next fractions.
                tau_dashes = torch.rand(
                    batch_size, N_dash, dtype=observation_embeddings.dtype, device=observation_embeddings.device
                )

                # Calculate quantile values of next observations and next actions.
                next_sa_quantiles = evaluate_quantile_at_action(
                    target_net.calculate_quantiles(tau_dashes, observation_embeddings=next_observation_embeddings),
                    next_actions,
                ).transpose(1, 2)

                # Calculate target quantile values.
                target_sa_quantiles = b_rewards[..., None] + (1.0 - b_dones[..., None]) * gamma * next_sa_quantiles

            td_errors = target_sa_quantiles - current_sa_quantiles
            # Calculate huber loss element-wisely.
            element_wise_huber_loss = torch.where(
                td_errors.abs() <= kappa, 0.5 * td_errors.pow(2), kappa * (td_errors.abs() - 0.5 * kappa)
            )

            # Calculate quantile huber loss element-wisely.
            element_wise_quantile_huber_loss = (
                torch.abs(taus[..., None] - (td_errors.detach() < 0).float()) * element_wise_huber_loss / kappa
            )

            # Quantile huber loss.
            batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
            quantile_huber_loss = batch_quantile_huber_loss.mean()

            optimizer.zero_grad()
            quantile_huber_loss.backward()
            optimizer.step()

        if global_step % eval_interval == 0:
            online_net.train()

    print(f"Episode: {episodes:<4}  " f"episode steps: {episode_steps:<4}  " f"return: {episode_return:<5.1f}")
    wandb.log(dict(global_step=global_step, episodic_return=episode_return))

    if global_step > num_steps:
        break
env.close()
