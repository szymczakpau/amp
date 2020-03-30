from typing import Any
from typing import Dict
from typing import Optional

from keras import layers
from keras import models
from keras import optimizers

from amp.models.discriminators import discriminator
from amp.models import model


class VeltriAMPClassifier(discriminator.Discriminator):
    """
    The discriminator part of cVAE.
    AMP classifier based on Veltri et al.(2018)
    """
    DEFAULT_INPUT_SHAPE = (21, 100)

    def __init__(
            self,
            embedding: layers.Embedding,
            convolution: layers.Layer,
            lstm: layers.Layer,
            dense_output: layers.Layer,
            name: str = 'VeltriAMPClassifier',
    ):
        self.embedding = embedding
        self.convolution = convolution
        self.lstm = lstm
        self.dense_output = dense_output
        self.name = name

        # TBD
        self.dense_emb = layers.Dense(self.embedding.output_dim, use_bias=False)

    def output_tensor(self, input_=None):

        if input_ is None:
            x = layers.Input(shape=self.DEFAULT_INPUT_SHAPE)
        else:
            x = input_

        emb = self.embedding(x)
        conv = self.convolution(emb)
        pool = layers.MaxPooling1D(pool_size=5)(conv)
        lstm = self.lstm(pool)
        return self.dense_output(lstm)

    def output_tensor_with_dense_input(self, input_: Optional[Any]):
        if input_ is None:
            x = layers.Input(shape=(self.DEFAULT_INPUT_SHAPE[1], 20))
        else:
            x = input_

        emb = self.dense_emb(x)
        self.dense_emb.set_weights(self.embedding.get_weights())
        self.dense_emb.trainable = self.embedding.trainable

        conv = self.convolution(emb)

        pool = layers.MaxPooling1D(pool_size=5)(conv)
        lstm = self.lstm(pool)
        return self.dense_output(lstm)

    def __call__(self, input_=None):
        x = input_ if input_ is not None else layers.Input(shape=(self.DEFAULT_INPUT_SHAPE[1],))
        model = models.Model(x, self.output_tensor(x))
        adam = optimizers.Adam(lr=0.01)

        model.compile(
            loss='binary_crossentropy',
            metrics=['accuracy'],
            optimizer=adam
        )

        return model

    def freeze_layers(self):
        self.embedding.trainable = False
        self.convolution.trainable = False
        self.lstm.trainable = False

    def unfreeze_layers(self):
        self.embedding.trainable = True
        self.convolution.trainable = True
        self.lstm.trainable = True

    def get_config_dict(self) -> Dict:
        return {
            'type': type(self).__name__,
            'name': self.name,
        }

    def get_layers_with_names(self) -> Dict[str, layers.Layer]:
        return {

            f'{self.name}_embedding': self.embedding,
            f'{self.name}_lstm': self.lstm,
            f'{self.name}_convolution': self.convolution,
            f'{self.name}_dense_output': self.dense_output,
        }

    @classmethod
    def from_config_dict_and_layer_collection(
            cls,
            config_dict: Dict,
            layer_collection: model.ModelLayerCollection,
    ) -> "VeltriAMPClassifier":
        return cls(
            name=config_dict['name'],
            embedding=layer_collection[config_dict['name'] + '_embedding'],
            convolution=layer_collection[config_dict['name'] + '_convolution'],
            lstm=layer_collection[config_dict['name'] + '_lstm'],
            dense_output=layer_collection[config_dict['name'] + '_dense_output'],
        )


class VeltriAMPClassifierFactory:

    @staticmethod
    def get_default() -> VeltriAMPClassifier:

        emb = layers.Embedding(
            input_dim=20,
            output_dim=64,
            input_length=100,
        )
        conv = layers.Convolution1D(
            filters=64,
            kernel_size=16,
            padding='same',
            strides=1,
            activation='relu'
        )
        lstm = layers.LSTM(
            100,
            unroll=True,
            stateful=False,
            dropout=0.1)
        dense_output = layers.Dense(1, activation='sigmoid')
        return VeltriAMPClassifier(
            embedding=emb,
            convolution=conv,
            lstm=lstm,
            dense_output=dense_output,
        )
