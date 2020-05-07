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
            encoder: amp_encoder.Encoder,
            decoder: amp_decoder.Decoder,
            discriminator: amp_discriminator,
            kl_weight: float,
            master_optimizer: optimizers.Optimizer,
    ):

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.kl_weight = kl_weight
        self.master_optimizer = master_optimizer

    @staticmethod
    def _zero_loss(y_true, y_pred):
        return tf.constant(0.0)

    def build(self, input_shape: Optional):
        inputs = layers.Input(shape=(input_shape[0],))
        z_mean, z_sigma, z = self.encoder.output_tensor(inputs)
        amp_in = layers.Input(shape=(1,))
        z_cond = layers.concatenate([z, amp_in])
        reconstructed = self.decoder.output_tensor(z_cond)
        y = vae_loss.VAELoss(
            kl_weight=self.kl_weight,
        )([inputs, reconstructed, z_mean, z_sigma])
        discriminator_output = self.discriminator.output_tensor_with_dense_input(
            input_=reconstructed,
        )
        self.discriminator.freeze_layers()

        vae = models.Model(
            inputs=[inputs, amp_in],
            outputs=[discriminator_output, y]
            )

        kl_metric = metrics.kl_loss(z_mean, z_sigma)

        def _kl_metric(y_true, y_pred):
            return kl_metric

        reconstruction_acc = metrics.sparse_categorical_accuracy(inputs, reconstructed)

        def _reconstruction_acc(y_true, y_pred):
          return reconstruction_acc

        rcl = metrics.reconstruction_loss(inputs, reconstructed)

        def _rcl(y_true, y_pred):
          return rcl

        amino_acc, empty_acc = metrics.get_generation_acc()(inputs, reconstructed)

        def _amino_acc(y_true, y_pred):
          return amino_acc

        def _empty_acc(y_true, y_pred):
          return empty_acc

        vae.compile(
            optimizer=self.master_optimizer,
            loss=['binary_crossentropy', 'mae'],
            loss_weights=[0.3, 1.0],
            metrics=[
                ['acc', 'binary_crossentropy'],
                [_kl_metric, _rcl, _reconstruction_acc, _amino_acc, _empty_acc]
                ]
        )
        return vae