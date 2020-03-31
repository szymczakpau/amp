from keras import layers
from keras import optimizers
from keras import models

from amp.models.decoders import decoder as amp_decoder
from amp.models.discriminators import discriminator as amp_discriminator
from amp.models.encoders import encoder as amp_encoder
from amp.layers import vae_loss
from amp.utils import metrics


class WakeModel:

    DEFAULT_SHAPE = (20, 100)

    def __init__(
            self,
            kl_weight: float,
            optimizer: optimizers.Optimizer,
            discriminator: amp_discriminator.Discriminator,
            decoder: amp_decoder.Decoder,
            encoder: amp_encoder.Encoder,
    ):
        self.kl_weight = kl_weight
        self.optimizer = optimizer
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def call(self, inputs=None):
        inputs = inputs if inputs is not None else layers.Input(shape=(self.DEFAULT_SHAPE[1],))
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

        return vae


class WakeModelsFactory:

    @staticmethod
    def get_default(
            lr: float,
            kl_weight: float,
            discriminator: amp_discriminator.Discriminator,
            decoder: amp_decoder.Decoder,
            encoder: amp_encoder.Encoder,
    ):
        optimizer = optimizers.Adam(lr=lr)
        return WakeModel(
            optimizer=optimizer,
            kl_weight=kl_weight,
            discriminator=discriminator,
            decoder=decoder,
            encoder=encoder,
        )
