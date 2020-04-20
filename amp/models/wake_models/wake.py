from abc import ABC
from typing import Any
from typing import Optional

from amp.models import model


class WakeModel(model.Model, ABC):
    """ Model with wake phase of VAE """

    def __call__(self, input_: Optional[Any] = None):
        raise NotImplementedError

