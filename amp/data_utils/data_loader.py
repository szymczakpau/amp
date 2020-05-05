from keras.preprocessing import sequence
from keras.preprocessing import text
from sklearn import model_selection

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

import pandas as pd
import numpy as np

from amp.data_utils import fasta

VOCAB_SIZE = 20

class ClassifierDataManager():

    def __init__(
            self,
            positive_file,
            negative_file,
            min_len,
            max_len,
            test_size: int = 0.1,
            val_size: int = 0.1,
            vocab_size: int = VOCAB_SIZE,
    ):
        self.positive_file = positive_file
        self.negative_file = negative_file
        self.positive_data = None
        self.negative_data = None
        self.min_len = min_len
        self.max_len = max_len
        self.test_size = test_size + val_size
        self.val_size = val_size / self.test_size
        self.vocab_size = vocab_size

        self.data_loader = ClassifierDataLoader(
            self.positive_file,
            self.negative_file,
        )

        self.data_filter = None
        self.data_equalizer = None
        self.data_splitter = None

    def load_data(self):
        self.positive_data, self.negative_data = self.data_loader.get_data()
        return self.positive_data, self.negative_data

    def filter_data(self):
        self.data_filter = ClassifierDataFilter(
            self.positive_data,
            self.negative_data,
            self.min_len,
            self.max_len
        )

        self.positive_data, self.negative_data = self.data_filter.filter_by_length()
        return self.positive_data, self.negative_data

    def equalize_data(self):
        self.data_equalizer = ClassifierDataEqualizer(
            self.positive_data,
            self.negative_data,
        )
        self.positive_data, self.negative_data = self.data_equalizer.balance()
        return self.positive_data, self.negative_data

    def plot_distributions(self):
        positive_seq = self.positive_data['Sequence'].tolist()
        positive_lengths = [len(seq) for seq in positive_seq]

        negative_seq = self.negative_data['Sequence'].tolist()
        negative_lengths = [len(seq) for seq in negative_seq]

        fig, (ax2, ax3) = plt.subplots(figsize=(12, 6), ncols=2)
        sns.distplot(positive_lengths, ax=ax2)
        sns.distplot(negative_lengths, ax=ax3)
        ax2.set_title("Positive")
        ax3.set_title("Negative")

        plt.show()

    def split_data(self):
        self.data_splitter = ClassifierDataSplitter(
            self.positive_data,
            self.negative_data,
            self.max_len,
            self.test_size,
            self.val_size,
            self.vocab_size
        )
        x_train, x_test, x_val, y_train, y_test, y_val = self.data_splitter.get_train_test_val()
        return x_train, x_test, x_val, y_train, y_test, y_val

    def get_classifier_data(self):
        self.load_data()
        self.filter_data()
        self.equalize_data()
        x_train, x_test, x_val, y_train, y_test, y_val = self.split_data()
        return x_train, x_test, x_val, y_train, y_test, y_val


class ClassifierDataLoader():
    """Load positive and negative data set"""

    def __init__(
            self,
            positive_file: str,
            negative_file: str,

    ):
        self.positive_data = None
        self.negative_data = None
        self.positive_file = positive_file
        self.negative_file = negative_file

    def _load_data(self):
        self.positive_data = pd.read_csv(self.positive_file)
        self.negative_data = pd.read_csv(self.negative_file)

    def load_data(self):
        if self.positive_data is None or self.negative_data is None:
            self._load_data()
        else:
            pass

    def get_data(self):
        self.load_data()
        return (self.positive_data, self.negative_data)


class ClassifierDataFilter():
    """Filter data by length"""

    def __init__(
            self,
            positive_data,
            negative_data,
            min_len,
            max_len
    ):
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.min_len = min_len
        self.max_len = max_len

    def _filter_by_length(self, df):
        mask = (df['Sequence'].str.len() >= self.min_len) & (df['Sequence'].str.len() <= self.max_len)
        return df.loc[mask]

    def filter_by_length(self):
        self.positive_data = self._filter_by_length(self.positive_data)
        return self.positive_data, self.negative_data


import random


class ClassifierDataEqualizer():
    """Balance the distributions between the datasets"""

    def __init__(
            self,
            positive_data,
            negative_data,
    ):

        self.positive_data = positive_data
        self.negative_data = negative_data

    def _get_probs(self, lengths):
        probs = {}
        count = 0
        for l in lengths:
            count += 1
            if l in probs:
                probs[l] += 1
            else:
                probs[l] = 0
        probs = {k: round(v / count, 4) for k, v in probs.items()}
        return probs

    def _draw_subsequences(self, df, new_lengths):

        new_lengths.sort(reverse=True)
        df = df.sort_values(by="Sequence length", ascending=False)

        d = []
        for row, new_length in zip(df.itertuples(), new_lengths):
            seq = row[2]
            curr_length = row[3]
            if new_length > curr_length:
                new_seq = seq
            elif new_length == curr_length:
                new_seq = seq
            else:
                begin = random.randrange(0, int(curr_length) - new_length)
                new_seq = seq[begin:begin + new_length]
            d.append(
                {
                    'Name': row[1],
                    'Sequence': new_seq,
                }
            )
        new_df = pd.DataFrame(d)
        return new_df

    def balance_distributions(self):

        positive_seq = self.positive_data['Sequence'].tolist()
        positive_lengths = [len(seq) for seq in positive_seq]

        negative_seq = self.negative_data['Sequence'].tolist()
        negative_lengths = [len(seq) for seq in negative_seq]
        self.negative_data.loc[:, "Sequence length"] = negative_lengths

        probs = self._get_probs(positive_lengths)
        new_negative_lengths = random.choices(list(probs.keys()), probs.values(), k=len(negative_lengths))
        self.negative_data = self._draw_subsequences(self.negative_data, new_negative_lengths)
        return self.positive_data, self.negative_data

    def balance(self):
        self.balance_distributions()
        return self.positive_data, self.negative_data


class ClassifierDataSplitter():
    """Merge the datasets and split into train, test, val"""

    def __init__(
            self,
            positive_data,
            negative_data,
            max_len,
            test_size,
            val_size,
            vocab_size
    ):
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.merged = None
        self.max_len = max_len
        self.test_size = test_size
        self.val_size = val_size
        self.vocab_size = vocab_size

    def join_datasets(self):
        self.positive_data.loc[:, 'Label'] = 1
        self.negative_data.loc[:, 'Label'] = 0
        self.merged = pd.concat([self.positive_data, self.negative_data])
        return self.merged

    def split(self, x, y):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=1,
        )

        x_test, x_val, y_test, y_val = model_selection.train_test_split(
            x_test,
            y_test,
            test_size=self.val_size,
            random_state=1,
        )

        return (x_train, x_test, x_val, y_train, y_test, y_val)

    def _to_one_hot(self, x):
        alphabet = list('ACDEFGHIKLMNPQRSTVWY')
        classes = range(1, 21)
        aa_encoding = dict(zip(alphabet, classes))

        return [[aa_encoding[aa] for aa in seq] for seq in x]

    def _pad(self, x):
        return sequence.pad_sequences(
            x,
            maxlen=self.max_len,
            padding='post',
            value=0.0
        )

    def get_train_test_val(self):
        self.join_datasets()

        x = np.asarray(self.merged['Sequence'].tolist())
        y = np.asarray(self.merged['Label'].tolist())

        x = self._pad(self._to_one_hot(x))
        x_train, x_test, x_val, y_train, y_test, y_val = self.split(x, y)

        return x_train, x_test, x_val, y_train, y_test, y_val
