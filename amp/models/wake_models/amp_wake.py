from keras import layers
from keras import optimizers
from keras import models

from amp.models.wake_models import wake
from amp.models.decoders import decoder as amp_decoder
from amp.models.discriminators import discriminator as amp_discriminator
from amp.models.encoders import encoder as amp_encoder
from amp.layers import vae_loss
from amp.utils import metrics


class AMPWakeModel(wake.WakeModel):

    def __init__(
            self,
            kl_weight: float,
            input_shape: tuple,
            optimizer: optimizers.Optimizer,
            discriminator: amp_discriminator.Discriminator,
            decoder: amp_decoder.Decoder,
            encoder: amp_encoder.Encoder,
            name: str = 'AMPWakeModel'
    ):
        self.kl_weight = kl_weight
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.name = name

    def call(self, inputs=None):
        inputs = inputs if inputs is not None else layers.Input(shape=(self.input_shape[0],))
        z_mean, z_sigma, z = self.encoder.output_tensor(inputs)
        disc_out = self.discriminator.output_tensor(inputs)
        z_cond = layers.concatenate([z, disc_out])
        reconstructed = self.decoder.output_tensor(z_cond)
        y = vae_loss.VAELoss(kl_weight=self.kl_weight)([inputs, reconstructed, z_mean, z_sigma])

        vae = models.Model(inputs, y)

        vae.compile(
            optimizer=self.optimizer,
            loss=None,
        )

        vae.metrics_tensors.append(metrics.kl_loss(z_mean, z_sigma))
        vae.metrics_names.append("kl")

        vae.metrics_tensors.append(metrics.sparse_categorical_accuracy(inputs, reconstructed))
        vae.metrics_names.append("acc")

        vae.metrics_tensors.append(metrics.reconstruction_loss(inputs, reconstructed))
        vae.metrics_names.append("rcl")

        amino_acc, empty_acc = metrics.get_generation_acc()(inputs, reconstructed)

        vae.metrics_tensors.append(amino_acc)
        vae.metrics_names.append("amino_acc")

        vae.metrics_tensors.append(empty_acc)
        vae.metrics_names.append("empty_acc")

        return vae


class WakeModelFactory:

    @staticmethod
    def get_default(
            lr: float,
            kl_weight: float,
            max_length: int,
            discriminator: amp_discriminator.Discriminator,
            decoder: amp_decoder.Decoder,
            encoder: amp_encoder.Encoder,
    ):
        optimizer = optimizers.Adam(lr=lr)
        return AMPWakeModel(
            optimizer=optimizer,
            kl_weight=kl_weight,
            input_shape=(max_length, 20),
            discriminator=discriminator,
            decoder=decoder,
            encoder=encoder,
        )
