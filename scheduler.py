"""

This module implements the noise scheduler.

"""


from abc import ABC, abstractmethod
import torch
from torch import Tensor

class KappaScheduler(ABC):
    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor: ...
    @abstractmethod
    def derivative(self, t: Tensor) -> Tensor: ...

class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 1.0, b: float = 1.0):
        self.a = a; self.b = b
    def __call__(self, t: Tensor) -> Tensor:
        return -2*t**3 + 3*t**2 + self.a*(t**3 - 2*t**2 + t) + self.b*(t**3 - t**2)
    def derivative(self, t: Tensor) -> Tensor:
        return -6*t**2 + 6*t + self.a*(3*t**2 - 4*t + 1) + self.b*(3*t**2 - 2*t)