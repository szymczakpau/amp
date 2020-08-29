import numpy as np

from keras.callbacks import Callback
from keras import backend

from amp.data_utils import sequence


class VAECallback(Callback):
    def __init__(
            self,
            encoder,
            decoder,
            classifier,
            kl_annealrate,
            max_kl,
            kl_weight,
            tau,
            tau_annealrate,
            min_tau,
            max_length:int=25,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.kl_annealrate = kl_annealrate
        self.max_kl = max_kl
        self.kl_weight = kl_weight
        self.tau = tau
        self.tau_annealrate = tau_annealrate
        self.min_tau = min_tau
        self.positive_callback_sample = sequence.pad(sequence.to_one_hot(['GFKDLLKGAAKALVKTVLF'])).reshape(1, max_length)
        self.negative_callback_sample = sequence.pad(sequence.to_one_hot(['FPSELANMKNALGFFHIGEIF'])).reshape(1, max_length)

    def on_epoch_end(self, epoch, logs={}):
        alphabet = list('ACDEFGHIKLMNPQRSTVWY')
        new_kl = np.min([backend.get_value(self.kl_weight) * np.exp(self.kl_annealrate * epoch), self.max_kl])
        backend.set_value(self.kl_weight, new_kl)

        pos_encoded_sample = self.encoder.predict(self.positive_callback_sample)
        neg_encoded_sample = self.encoder.predict(self.negative_callback_sample)
        pos_sample = np.concatenate([pos_encoded_sample, np.array([[1]])], axis=1)
        neg_sample = np.concatenate([neg_encoded_sample, np.array([[0]])], axis=1)
        pos_prediction = self.decoder.predict(pos_sample)
        neg_prediction = self.decoder.predict(neg_sample)
        pos_peptide = ''.join([alphabet[el - 1] if el != 0 else "'" for el in pos_prediction[0].argmax(axis=1)])
        neg_peptide = ''.join([alphabet[el - 1] if el != 0 else "'" for el in neg_prediction[0].argmax(axis=1)])
        pos_class_prob = self.classifier.predict(np.array([pos_prediction[0].argmax(axis=1)]))
        neg_class_prob = self.classifier.predict(np.array([neg_prediction[0].argmax(axis=1)]))

        new_tau = np.max([backend.get_value(self.tau) * np.exp(- self.tau_annealrate * epoch), self.min_tau])
        backend.set_value(self.tau, new_tau)

        print(
            f'Original positive: GGAGHVPEYFVGIGTPISFYG, \ngenerated: {pos_peptide}, \nAMP probability: {pos_class_prob[0][0]}')
        print(
            f'Original negative: FPSELANMKNALGFFHIGEIF, \ngenerated: {neg_peptide}, \nAMP probability: {neg_class_prob[0][0]}')

        print("Current KL weight is " + str(backend.get_value(self.kl_weight)))
        print("Current temperature is " + str(backend.get_value(self.tau)))