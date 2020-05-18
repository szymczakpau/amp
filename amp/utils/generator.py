import csv

from keras.preprocessing import sequence
from keras.preprocessing import text
import numpy as np

VOCAB_SIZE = 20
MAX_LENGTH = 100


class UniprotGenerator:

    def __init__(
            self,
            path,
            batch_size,
            nb_of_seqs_in_file: int,
            max_len: int = MAX_LENGTH,
            vocab_size: int = VOCAB_SIZE,
    ):
        self.nb_of_seqs_in_file = nb_of_seqs_in_file
        self.path = path
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_len = max_len

    def __len__(self):
        return int(np.floor(self.nb_of_seqs_in_file / self.batch_size))

    def __iter__(self):
        """Generate one batch of data"""
        # Generate indices of the batch
        while True:
            current_file_index = 0
            with open(self.path, 'r') as csv_file:
                reader = csv.reader(csv_file)
                current_batch = []
                current_batch_index = 0
                for line_number, row in enumerate(reader):
                    if line_number == 0:
                        continue
                    current_batch.append(row[1])  # Only sequence
                    current_batch_index += 1
                    current_file_index += 1
                    if current_batch_index == self.batch_size:
                        one_hot_batch = self.to_one_hot(current_batch)
                        padded_batch = self.pad(one_hot_batch)
                        yield padded_batch
                        current_batch_index = 0
                        current_batch = []
                        if current_file_index + self.batch_size > self.nb_of_seqs_in_file:
                            break

    def to_one_hot(self, x):
        alphabet = list('ACDEFGHIKLMNPQRSTVWY')
        classes = range(1, 21)
        aa_encoding = dict(zip(alphabet, classes))

        return [[aa_encoding[aa] for aa in seq] for seq in x]

    def pad(self, x):
        return sequence.pad_sequences(
            x,
            maxlen=self.max_len,
            padding='post',
            value=0.0
        )
