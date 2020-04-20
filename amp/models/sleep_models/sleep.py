from abc import ABC
from typing import Any
from typing import Optional

from amp.models import model


class SleepModel(model.Model, ABC):
    """ Model with sleep phase of VAE """

    def __call__(self, input_: Optional[Any] = None):
        raise NotImplementedError

