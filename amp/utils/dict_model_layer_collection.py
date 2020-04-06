from typing import Dict
import json
import os

from keras import layers
import numpy as np

from amp.models import model


class DictModelLayerCollection(model.ModelLayerCollection):

    LAYER_CONFIG_NAME = 'layer.json'
    WEIGHT_NAME = '_weight.npy'

    def __init__(self, dict_to_layer: Dict[str, layers.Layer]):
        self._dict_to_layer = dict_to_layer

    def __add__(self, layers_with_names: Dict[str, layers.Layer]):
        for layer_name, layer in layers_with_names.items():
            self._dict_to_layer[layer_name] = layer

    def __getitem__(self, item: str) -> layers.Layer:
        return self._dict_to_layer[item]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for current_layer_name, current_layer in self._dict_to_layer.items():
            current_layer_path = os.path.join(path, current_layer_name)
            self._save_single_layer(path=current_layer_path, layer=current_layer)

    def _save_single_layer(self, path: str, layer: layers.Layer):
        os.makedirs(path, exist_ok=True)
        config_path = os.path.join(path, self.LAYER_CONFIG_NAME)
        with open(config_path, 'w') as json_handle:
            config = {
                'type': type(layer).__name__,
                'layer_config': layer.get_config(),
            }
            json.dump(config, json_handle)
            for layer_nb, weight in enumerate(layer.get_weights()):
                current_weight_path = os.path.join(path, f'{layer_nb}{self.WEIGHT_NAME}')
                np.save(current_weight_path, weight)
