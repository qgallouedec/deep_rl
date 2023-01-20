from collections import deque
from typing import Any, Dict, Tuple

import gym
import numpy as np
import torch
from gym import spaces

try:
    import cv2  # pytype:disable=import-error

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None


ATARI_IDS = [
    "AdventureNoFrameskip-v4",
    "AirRaidNoFrameskip-v4",
    "AlienNoFrameskip-v4",
    "AmidarNoFrameskip-v4",
    "AssaultNoFrameskip-v4",
    "AsterixNoFrameskip-v4",
    "AsteroidsNoFrameskip-v4",
    "AtlantisNoFrameskip-v4",
    "BankHeistNoFrameskip-v4",
    "BattleZoneNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "BerzerkNoFrameskip-v4",
    "BowlingNoFrameskip-v4",
    "BoxingNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "CarnivalNoFrameskip-v4",
    "CentipedeNoFrameskip-v4",
    "ChopperCommandNoFrameskip-v4",
    "CrazyClimberNoFrameskip-v4",
    "DefenderNoFrameskip-v4",
    "DemonAttackNoFrameskip-v4",
    "DoubleDunkNoFrameskip-v4",
    "ElevatorActionNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "FishingDerbyNoFrameskip-v4",
    "FreewayNoFrameskip-v4",
    "FrostbiteNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "GravitarNoFrameskip-v4",
    "HeroNoFrameskip-v4",
    "IceHockeyNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "JourneyEscapeNoFrameskip-v4",
    "KangarooNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "MontezumaRevengeNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "NameThisGameNoFrameskip-v4",
    "PhoenixNoFrameskip-v4",
    "PitfallNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "PooyanNoFrameskip-v4",
    "PrivateEyeNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "RiverraidNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "RobotankNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "SkiingNoFrameskip-v4",
    "SolarisNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "StarGunnerNoFrameskip-v4",
    "TennisNoFrameskip-v4",
    "TimePilotNoFrameskip-v4",
    "TutankhamNoFrameskip-v4",
    "UpNDownNoFrameskip-v4",
    "VentureNoFrameskip-v4",
    "VideoPinballNoFrameskip-v4",
    "WizardOfWorNoFrameskip-v4",
    "YarsRevengeNoFrameskip-v4",
    "ZaxxonNoFrameskip-v4",
]


class StickyActionWrapper(gym.Wrapper):
    """
    Sticky action.

    Args:
        env (gym.Env): Environment to wrap
        action_repeat_probability (float): Probability of repeating the previous action
    """

    def __init__(self, env: gym.Env, action_repeat_probability: float) -> None:
        super().__init__(env)
        self.action_repeat_probability = action_repeat_probability

    def reset(self) -> np.ndarray:
        self.previous_action = None
        return super().reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.previous_action is not None:
            if self.np_random.random() < self.action_repeat_probability:
                action = self.previous_action
        self.previous_action = action
        return self.env.step(action)


class NoopResetWrapper(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset. No-op is assumed to be action 0.

    Args:
        env (gym.Env): Environment to wrap
        noop_max (int): Maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self) -> np.ndarray:
        self.env.reset()
        noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


class FireResetWrapper(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.

    Args:
        env (gym.Env): Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self) -> np.ndarray:
        self.env.reset()
        _, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            obs = self.env.reset()
        return obs


class EpisodicLifeWrapper(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.

    Args:
        env (gym.Env): Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.inner_done = True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        self.inner_done = done
        # Check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # For Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        # Calls inner reset only when lives are exhausted.
        # This way all states are still reachable even though lives are episodic,
        # and the learner need not know about any of this behind-the-scenes.
        if self.inner_done:
            obs = self.env.reset()
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipWrapper(gym.Wrapper):
    """
    Return only every skip-th frame (frameskipping)

    Args:
        env (gym.Env): Environment to wrap
        frame_skip (int): Number of frame to skip
    """

    def __init__(self, env: gym.Env, frame_skip: int = 4) -> None:
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self.frame_skip = frame_skip

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # Repeat action, sum reward, and max over last observations.
        total_reward = 0.0
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info


class ClipRewardWrapper(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.

    Args:
        env (gym.Env): Environment to wrap
    """

    def reward(self, reward: float) -> float:
        return np.sign(reward)


class GrayscaleWrapper(gym.ObservationWrapper):
    """
    Convert observation to grayscale.

    Args:
        env (gym.Env): Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        height, width = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation[:, :, None]


class ResizeWrapper(gym.ObservationWrapper):
    """
    ResizeWrapper observation frame.

    Args:
        env (gym.Env): Environment to wrap
        width (int): New width
        height (int): New height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        nb_channels = self.observation_space.shape[2]
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, nb_channels), dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation = cv2.resize(observation, (self.width, self.height), interpolation=cv2.INTER_AREA)
        observation = observation.reshape(self.observation_space.shape)  # prevent squeezing when observation is grayscale
        return observation


class AtariWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings.

    Includes:
        - Sticky action
        - Noop reset
        - Fire reset
        - Maxpool and frame skip
        - Episodic life
        - Grayscale and resize
        - Clip reward

    Args:
        env (gym.Env): Environment to wrap
        noop_max (int): Maximum value of no-ops to run. Defaults to 30.
        action_repeat_probability (float): Probability of repeating the previous action. Defaults to 0.25.
        frame_skip (int): Number of frame to skip. Defaults to 4.
        width (int): Frame width. Defaults to 84.
        height (int): Frame height. Defaults to 84.
        terminal_on_life_loss (bool): Whether to only reset when all lives are exausted. Defaults to True.
        clip_reward (bool): Whether to clip the reward. Defaults to True.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        action_repeat_probability: int = 0.25,
        frame_skip: int = 4,
        width: int = 84,
        height: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionWrapper(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetWrapper(env, noop_max=noop_max)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetWrapper(env)
        if frame_skip > 1:
            env = MaxAndSkipWrapper(env, frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeWrapper(env)
        env = GrayscaleWrapper(env)
        env = ResizeWrapper(env, width, height)
        if clip_reward:
            env = ClipRewardWrapper(env)
        super().__init__(env)
