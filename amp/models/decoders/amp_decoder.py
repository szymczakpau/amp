from keras import layers
from keras import models
import tensorflow as tf

from amp.layers import autoregressive_gru, gumbel_softmax
from amp.models.decoders import decoder


class AMPDecoder(decoder.Decoder):

    def __init__(
            self,
            latent_dim: int,
            dense: layers.Dense,
            recurrent_autoregressive: layers.Recurrent,
            activation: layers.Layer,
    ):
        self.latent_dim = latent_dim
        self.latent_to_hidden = recurrent_autoregressive
        self.dense = dense
        self.activation = activation

    def output_tensor(self, input_=None):
        z = input_
        latent_to_hidden = self.latent_to_hidden(z)
        dense = self.dense(latent_to_hidden)
        return layers.TimeDistributed(self.activation)(dense)

    def __call__(self, input_=None):
        z = input_
        generated_x = self.output_tensor(z)
        model = models.Model(z, generated_x)
        return model


class AMPDecoderFactory:

    @staticmethod
    def build_default(
            latent_dim: int,
            gumbel_temperature: tf.Variable,
    ):
        recurrent_autoregressive = autoregressive_gru.AutoregressiveGRU.build_for_gru(
            latent_dim,
            100,
        )
        dense = layers.Dense(20)
        return AMPDecoder(
            latent_dim=latent_dim,
            recurrent_autoregressive=recurrent_autoregressive,
            dense=dense,
            activation=gumbel_softmax.GumbelSoftmax(temperature=gumbel_temperature)
        )
