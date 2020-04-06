import json
import os

from amp.models import model as amp_model
from amp.models import model_garden
from amp.utils import dict_model_layer_collection


class BasicModelSerializer(amp_model.ModelSerializer):

    MODEL_CONFIG_NAME = 'model_config.json'
    LAYER_DIR_NAME = 'layers'

    def save_model(self, model: amp_model.Model, path: str):
        os.makedirs(path, exist_ok=True)
        model_config_path = os.path.join(path, self.MODEL_CONFIG_NAME)
        with open(model_config_path, 'w') as json_handle:
            json.dump(model.get_config_dict(), json_handle)
        model_layer_collection = dict_model_layer_collection.DictModelLayerCollection(
            dict_to_layer=model.get_layers_with_names(),
        )
        layer_saving_path = os.path.join(path, self.LAYER_DIR_NAME)
        model_layer_collection.save(layer_saving_path)

    def load_model(self, path: str) -> amp_model.Model:
        model_config_path = os.path.join(path, self.MODEL_CONFIG_NAME)
        with open(model_config_path, 'r') as json_handle:
            model_config = json.load(json_handle)
        model_class = model_garden.MODEL_GAREDN[model_config['type']]
        model_layer_collection_path = os.path.join(path, self.LAYER_DIR_NAME)
        return model_class.from_config_dict_and_layer_collection(
            config_dict=model_config,
            layer_collection=dict_model_layer_collection.DictModelLayerCollection.load(
                path=model_layer_collection_path,
            )
        )
