from typing import Optional

from keras import layers
from keras import models
from keras import optimizers

from amp.layers import vae_loss
from amp.models.encoders import encoder as amp_encoder
from amp.models.decoders import decoder as amp_decoder
from amp.models.discriminators import discriminator as amp_discriminator
from amp.utils import metrics


class DataGenerator:
    pass


class MasterAMPTrainer:

    def __init__(
            self,
            amp_data_generator: DataGenerator,
            unlabeled_data_generator: DataGenerator,
            encoder: amp_encoder.Encoder,
            decoder: amp_decoder.Decoder,
            discriminator: amp_discriminator,
            kl_weight: float,
            master_optimizer: optimizers.Optimizer,
    ):
        self.amp_data_generator = amp_data_generator
        self.unlabeled_data_generator = unlabeled_data_generator
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.kl_weight = kl_weight
        self.master_optimizer = master_optimizer

    def build(self, input_shape: Optional):
        inputs = layers.Input(shape=(input_shape[0],))
        z_mean, z_sigma, z = self.encoder.output_tensor(inputs)
        amp_in = layers.Input(shape=(1,))
        z_cond = layers.concatenate([z, amp_in])
        reconstructed = self.decoder.output_tensor(z_cond)
        y = vae_loss.VAELoss(
            kl_weight=self.kl_weight
        )([inputs, reconstructed, z_mean, z_sigma])
        discriminator_output = self.discriminator.output_tensor_with_dense_input(
            input=reconstructed,
        )
        vae = models.Model(inputs, discriminator_output)

        vae.compile(
            optimizer=self.master_optimizer,
            loss='binary_crossentropy',
            metrics=['acc', 'binary_crossentropy']
        )

        vae.metrics_tensors.append(metrics.kl_loss(z_mean, z_sigma))
        vae.metrics_names.append("kl")

        vae.metrics_tensors.append(
            metrics.sparse_categorical_accuracy(inputs, reconstructed),
        )
        vae.metrics_names.append("acc")

        vae.metrics_tensors.append(metrics.reconstruction_loss(inputs, reconstructed))
        vae.metrics_names.append("rcl")

        amino_acc, empty_acc = metrics.get_generation_acc()(inputs, reconstructed)

        vae.metrics_tensors.append(amino_acc)
        vae.metrics_names.append("amino_acc")

        vae.metrics_tensors.append(empty_acc)
        vae.metrics_names.append("empty_acc")

        return vae
