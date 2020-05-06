from typing import Dict

from keras import layers
from keras import models
import tensorflow as tf

from amp.layers import autoregressive_gru, gumbel_softmax
from amp.models.decoders import decoder
from amp.models import model


class AMPDecoder(decoder.Decoder):

    def __init__(
            self,
            latent_dim: int,
            dense: layers.Dense,
            recurrent_autoregressive: layers.Recurrent,
            activation: layers.Layer,
            name: str = 'AMPDecoder'
    ):
        self.latent_dim = latent_dim
        self.latent_to_hidden = recurrent_autoregressive
        self.dense = dense
        self.activation = activation
        self.name = name

    def output_tensor(self, input_=None):
        z = input_
        latent_to_hidden = self.call_layer_on_input(self.latent_to_hidden, z)
        dense = self.call_layer_on_input(self.dense, latent_to_hidden)
        return self.call_layer_on_input(layers.TimeDistributed(self.activation), dense)

    def __call__(self, input_=None):
        z = input_
        generated_x = self.output_tensor(z)
        model = models.Model(z, generated_x)
        return model

    def get_config_dict(self) -> Dict:
        return {
            'type': type(self).__name__,
            'name': self.name,
            'latent_dim': self.latent_dim,
        }

    def get_layers_with_names(self) -> Dict[str, layers.Layer]:
        return {
            f'{self.name}_latent_to_hidden': self.latent_to_hidden,
            f'{self.name}_dense': self.dense,
            f'{self.name}_activation': self.activation,
        }

    @classmethod
    def from_config_dict_and_layer_collection(
            cls,
            config_dict: Dict,
            layer_collection: model.ModelLayerCollection,
    ) -> "AMPDecoder":
        return cls(
            name=config_dict['name'],
            latent_dim=config_dict['latent_dim'],
            recurrent_autoregressive=layer_collection[config_dict['name'] + '_latent_to_hidden'],
            dense=layer_collection[config_dict['name'] + '_dense'],
            activation=layer_collection[config_dict['name'] + '_activation'],
        )
class AMPDecoderFactory:

    @staticmethod
    def build_default(
            latent_dim: int,
            gumbel_temperature: tf.Variable,
            max_length: int,
    ):
        recurrent_autoregressive = autoregressive_gru.AutoregressiveGRU.build_for_gru(
            latent_dim,
            max_length,
        )
        dense = layers.Dense(21)
        return AMPDecoder(
            latent_dim=latent_dim,
            recurrent_autoregressive=recurrent_autoregressive,
            dense=dense,
            activation=gumbel_softmax.GumbelSoftmax(temperature=gumbel_temperature)
        )
