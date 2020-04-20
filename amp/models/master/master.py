import numpy as np

from amp.models import model
from amp.models import conditional_vae as cv


class MasterModel(model.Model):

    def __init__(
            self,
            labeled_dataset_train,
            labeled_dataset_val,
            prior_label,
            unlabled_dataset,
            unlabeled_prior,
            encoder,
            decoder,
            discriminator,
    ):
        # TODO add rates etc.
        self.labeled_dataset_train = labeled_dataset_train
        self.labeled_dataset_val = labeled_dataset_val
        self.prior_label = prior_label
        self.unlabeled_prior = unlabeled_prior
        self.unlabled_dataset = unlabled_dataset
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def train_base_conditional_vae(self, epochs: int, batch_size: int):
        cv.ConditionalVae(encoder=self.encoder, decoder=self.decoder)
        conditional_vae_dataset = self.unlabled_dataset & self.prior_label
        cv.train_on_dataset(
            epochs=epochs,
            batch_size=batch_size,
            dataset=conditional_vae_dataset,
        )

    def train_step(self):
        self.sleep_step()
        self.wake_step()

    def sleep_step(self):
        sleep_batch = self._sleep_data_generator()
        self.discriminator.sleep_train_generator(sleep_batch)

    def wake_step(self):
        pass

    def _imaginary_data_generator(self):
        while True:
            unlabeled_prior_sample = self.unlabeled_prior.next() # z
            prior_sample = self.prior_label.next() # c
            slept_example = self.decoder.predict([unlabeled_prior_sample, prior_sample])
            yield [slept_example, prior_sample]

    def _sleep_data_generator(self):
        while True:
            imaginary_data = next(self._imaginary_data_generator())
            true_data = next(self.labeled_dataset_train)
            yield np.concatenate(imaginary_data[0], true_data[0]), np.concatenate(imaginary_data[1] + true_data[1])
