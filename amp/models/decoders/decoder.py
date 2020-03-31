from typing import Any
from typing import Optional


class Decoder:
    def output_tensor(self, input_: Optional[Any] = None):
        raise NotImplementedError

    def __call__(self, input_: Optional[Any] = None):
        raise NotImplementedError
