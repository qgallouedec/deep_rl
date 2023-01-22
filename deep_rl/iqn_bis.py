from collections import deque

import cv2
import gym
import numpy as np
import torch
from gym import spaces
from torch import nn, optim

import wandb

cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        :param env: (Gym Environment) the environment to wrap
        :param noop_max: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame (frameskipping)
        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward,
                 done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        :param env: (Gym Environment) the environment to wrap
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few
            # frames so its important to keep lives > 0, so that we only reset
            # once the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        :param kwargs: Extra keywords passed to env.reset() call
        :return: ([int] or [float]) the first observation of the environment
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


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
            low=0, high=255, shape=(1, self.height, self.width), dtype=env.observation_space.dtype
        )

    def observation(self, frame):
        """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[None, :, :]


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        """
        clips the reward to {+1, 0, -1} by its sign.
        :param env: (Gym Environment) the environment
        """
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """
        Bin reward to {+1, 0, -1} by its sign.
        :param reward: (float)
        """
        return np.sign(reward)


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
            shape=(shp[0] * n_frames, shp[1], shp[2]),
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
        return np.concatenate(np.array(self.frames), axis=0)


class AtariWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFramePyTorch(env)
        env = ClipRewardEnv(env)
        env = FrameStackPyTorch(env, 4)
        super().__init__(env)


class LazyMultiStepMemory:
    def __init__(self, capacity, state_shape, device):
        self.capacity = capacity
        self.state_shape = state_shape
        self.device = device
        self.reset()

    def append(self, state, action, reward, next_state, done):
        self.state.append(state)
        self.next_state.append(next_state)
        self.action[self._p] = action
        self.reward[self._p] = reward
        self.done[self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        while len(self.state) > self.capacity:
            del self.state[0]
            del self.next_state[0]

    def reset(self):
        self.state = []
        self.next_state = []
        self.action = np.empty((self.capacity, 1), dtype=np.int64)
        self.reward = np.empty((self.capacity, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, 1), dtype=np.float32)
        self._n = 0
        self._p = 0

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self.state), size=batch_size)
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty((batch_size, *self.state_shape), dtype=np.uint8)
        next_states = np.empty((batch_size, *self.state_shape), dtype=np.uint8)

        for i, index in enumerate(indices):
            _index = np.mod(index + bias, self.capacity)
            states[i, ...] = self.state[_index]
            next_states[i, ...] = self.next_state[_index]

        states = torch.ByteTensor(states).to(self.device).float() / 255.0
        next_states = torch.ByteTensor(next_states).to(self.device).float() / 255.0
        actions = torch.LongTensor(self.action[indices]).to(self.device)
        rewards = torch.FloatTensor(self.reward[indices]).to(self.device)
        dones = torch.FloatTensor(self.done[indices]).to(self.device)

        return states, actions, rewards, next_states, dones


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

    def forward(self, states):
        return self.net(states)


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

    def forward(self, state_embeddings, tau_embeddings):

        # NOTE: Because variable taus correspond to either \tau or \hat \tau
        # in the paper, N isn't neccesarily the same as fqf.N.
        batch_size = state_embeddings.shape[0]
        N = tau_embeddings.shape[1]

        # Reshape into (batch_size, 1, embedding_dim).
        state_embeddings = state_embeddings.view(batch_size, 1, self.embedding_dim)

        # Calculate embeddings of states and taus.
        embeddings = (state_embeddings * tau_embeddings).view(batch_size * N, self.embedding_dim)

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

    def calculate_state_embeddings(self, states):
        return self.dqn_net(states)

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):
        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, states=None, state_embeddings=None):
        batch_size = states.shape[0] if states is not None else state_embeddings.shape[0]
        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)
        taus = torch.rand(batch_size, self.K, dtype=state_embeddings.dtype, device=state_embeddings.device)
        quantiles = self.calculate_quantiles(taus, state_embeddings=state_embeddings)
        return quantiles.mean(dim=1)


def evaluate_quantile_at_action(s_quantiles, actions):
    batch_size = s_quantiles.shape[0]
    N = s_quantiles.shape[1]
    action_index = actions[..., None].expand(batch_size, N, 1)
    return s_quantiles.gather(dim=2, index=action_index)


class IQNAgent:
    def __init__(
        self,
        env,
        num_steps=2_500_000,
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
        epsilon_eval=0.001,
        epsilon_decay_steps=250_000,
        eval_interval=250_000,
        num_eval_steps=125_000,
        max_episode_steps=27_000,
        seed=0,
    ):
        self.env = env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.env.observation_space.seed(seed)
        self.env.action_space.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Replay memory which is memory-efficient to store stacked frames.
        self.memory = LazyMultiStepMemory(memory_size, self.env.observation_space.shape, self.device)
        self.global_step = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_actions = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.gamma_n = gamma
        self.start_steps = start_steps

        self._steps = 0
        self.epsilon_decay_steps = epsilon_decay_steps
        self.slope = (epsilon_train - 1.0) / self.epsilon_decay_steps

        self.epsilon_eval = epsilon_eval
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.max_episode_steps = max_episode_steps

        # Online network.
        self.online_net = IQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions,
            K=K,
            num_cosines=num_cosines,
        ).to(self.device)
        # Target network.
        self.target_net = IQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions,
            K=K,
            num_cosines=num_cosines,
        ).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.target_net.load_state_dict(self.online_net.state_dict())
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optim = optim.Adam(self.online_net.parameters(), lr=lr, eps=1e-2 / batch_size)

        self.N = N
        self.N_dash = N_dash
        self.K = K
        self.num_cosines = num_cosines
        self.kappa = kappa

    def learn(self):
        self.learning_steps += 1

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Calculate features of states.
        state_embeddings = self.online_net.calculate_state_embeddings(states)

        # Sample fractions.
        taus = torch.rand(self.batch_size, self.N, dtype=state_embeddings.dtype, device=state_embeddings.device)

        # Calculate quantile values of current states and actions at tau_hats.
        current_sa_quantiles = evaluate_quantile_at_action(
            self.online_net.calculate_quantiles(taus, state_embeddings=state_embeddings), actions
        )

        with torch.no_grad():
            # Calculate Q values of next states.
            next_state_embeddings = self.target_net.calculate_state_embeddings(next_states)
            next_q = self.target_net.calculate_q(state_embeddings=next_state_embeddings)

            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)

            # Sample next fractions.
            tau_dashes = torch.rand(self.batch_size, self.N_dash, dtype=state_embeddings.dtype, device=state_embeddings.device)

            # Calculate quantile values of next states and next actions.
            next_sa_quantiles = evaluate_quantile_at_action(
                self.target_net.calculate_quantiles(tau_dashes, state_embeddings=next_state_embeddings), next_actions
            ).transpose(1, 2)

            # Calculate target quantile values.
            target_sa_quantiles = rewards[..., None] + (1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles

        td_errors = target_sa_quantiles - current_sa_quantiles
        # Calculate huber loss element-wisely.
        element_wise_huber_loss = torch.where(
            td_errors.abs() <= self.kappa, 0.5 * td_errors.pow(2), self.kappa * (td_errors.abs() - 0.5 * self.kappa)
        )

        # Calculate quantile huber loss element-wisely.
        element_wise_quantile_huber_loss = (
            torch.abs(taus[..., None] - (td_errors.detach() < 0).float()) * element_wise_huber_loss / self.kappa
        )

        # Quantile huber loss.
        batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
        quantile_huber_loss = batch_quantile_huber_loss.mean()

        self.optim.zero_grad()
        quantile_huber_loss.backward()
        self.optim.step()

    def __del__(self):
        self.env.close()

    def run(self):
        while True:
            self.online_net.train()
            self.target_net.train()

            self.episodes += 1
            episode_return = 0.0
            episode_steps = 0

            done = False
            state = self.env.reset()

            while (not done) and episode_steps <= self.max_episode_steps:
                if self.global_step < self.start_steps or np.random.rand() < self.slope * self._steps + 1.0:
                    action = self.env.action_space.sample()
                else:
                    state_ = torch.ByteTensor(np.array(state)).unsqueeze(0).to(self.device).float() / 255.0
                    with torch.no_grad():
                        action = self.online_net.calculate_q(states=state_).argmax().item()

                next_state, reward, done, _ = self.env.step(action)
                self.memory.append(state, action, reward, next_state, done)

                self.global_step += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

                self._steps = min(self.epsilon_decay_steps, self._steps + 1)

                if self.global_step % self.target_update_interval == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                if self.global_step % self.update_interval == 0 and self.global_step >= self.start_steps:
                    self.learn()

                if self.global_step % self.eval_interval == 0:
                    self.online_net.train()

            print(f"Episode: {self.episodes:<4}  " f"episode steps: {episode_steps:<4}  " f"return: {episode_return:<5.1f}")
            wandb.log(dict(global_step=self.global_step, episodic_return=episode_return))

            if self.global_step > self.num_steps:
                break


if __name__ == "__main__":
    # Create environments.
    env_id = "PongNoFrameskip-v4"
    seed = 0
    env = gym.make(env_id)
    env = AtariWrapper(env)
    wandb.init(project="IQN")

    # Create the agent and run.
    agent = IQNAgent(
        env=env,
        # seed=seed,
        # num_steps=50_000,
        # lr=5e-04,
        # target_update_interval=1_000,
        # start_steps=500,
        # epsilon_decay_steps=2_000,
        # eval_interval=2_000,
        # num_eval_steps=2_000,
    )
    agent.run()
