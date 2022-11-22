from typing import Any, Dict, Optional, Tuple, Union

import gym
import torch
from torch import Tensor


class TorchWrapper(gym.Wrapper):
    """
    Torch wrapper. Actions and observations are Tensors instead of arrays.
    """

    def __init__(self, env: gym.Env, device: Optional[Union[torch.device, str]] = None) -> None:
        super().__init__(env)
        self.device = device

    def step(self, action: Tensor) -> Tuple[Tensor, float, bool, Dict[str, Any]]:
        action = action.cpu().numpy()
        observation, reward, done, info = self.env.step(action)
        return torch.tensor(observation).to(self.device), reward, done, info

    def reset(self) -> Tensor:
        observation = self.env.reset()
        return torch.tensor(observation).to(self.device)
