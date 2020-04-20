from keras import layers
from keras import optimizers
from keras import models
from keras import metrics

from amp.models.sleep_models import sleep
from amp.models.discriminators import discriminator as amp_discriminator
from amp.models.decoders import decoder as amp_decoder

class AMPSleepModel(sleep.SleepModel):

    def __init__(
            self,
            optimizer: optimizers.Optimizer,
            discriminator: amp_discriminator.Discriminator,
            decoder: amp_decoder.Decoder,
            name: str = 'AMPSleepModel'

    ):
        self.optimizer = optimizer
        self.discriminator = discriminator
        self.decoder = decoder
        self.name = name

    def call(self):
        z_input = layers.Input(shape=(self.decoder.latent_dim - 1,))
        c_input = layers.Input(shape=(1,))
        inputs = layers.concatenate([z_input, c_input])
        encoded = self.decoder.output_tensor(inputs)
        output = self.discriminator.output_tensor_with_dense_input(encoded)

        model = models.Model([z_input, c_input], output)
        model.add_loss(metrics.mean_squared_error(c_input, z_input))

        model.compile(
            optimizer=self.optimizer,
            loss=None
        )

        model.metrics_tensors.append(metrics.mean_squared_error(c_input, z_input))
        model.metrics_names.append("mse")

        return model


class SleepModelFactory:

    @staticmethod
    def get_default(
            lr: float,
            discriminator: amp_discriminator.Discriminator,
            decoder: amp_decoder.Decoder,
    ):
        optimizer = optimizers.Adam(lr=lr)
        return AMPSleepModel(
            optimizer=optimizer,
            discriminator=discriminator,
            decoder=decoder,
        )

