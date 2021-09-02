from abc import abstractmethod
from typing import Tuple

from torch import nn
import torch


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # The expected input shape of input
        self.expected_input = None
        # THe description of the model
        self.description = ""

    def set_description(self, description: str) -> None:
        """
        Set the description of the model
        :param description: str, the description
        :return: None
        """
        self.description = description

    def get_description(self) -> str:
        """
        Return the description of the model
        :return: str, the description of the model
        """
        return self.description

    def set_expected_input(self, shape: Tuple) -> None:
        """
        Set the shape of the expected input.
        :param shape: tuple, the shape
        :return: None
        """
        self.expected_input = shape

    def check_input(self, input_shape: Tuple) -> bool:
        """
        Check the input whether equals to the target shape
        :param input_shape: tuple, the input shape
        :return: bool, whether the two are equal
        """
        if self.shape is not None and input_shape == self.shape:
            return True
        return False
