from abc import abstractmethod

from torch import nn
import torch
import importlib


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.expected_input = None

    @abstractmethod
    def set_expected_input(self):
        pass

    @abstractmethod
    def check_input(self):
        pass

