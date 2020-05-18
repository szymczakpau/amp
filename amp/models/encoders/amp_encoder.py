from typing import Any
from typing import Dict
from typing import Optional

from keras import layers
from keras import models
from keras import backend

from amp.models.encoders import encoder
from amp.models import model


class AMPEncoder(encoder.Encoder):

    def __init__(
            self,
            embedding: layers.Embedding,
            hidden: layers.Layer,
            dense_z_mean: layers.Layer,
            dense_z_sigma: layers.Layer,
            input_shape: tuple,
            latent_dim: int,
            hidden_dim: int,
            name: str = 'AMPEncoder',
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape

        self.embedding = embedding
        self.hidden = hidden
        self.z_mean = dense_z_mean
        self.z_sigma = dense_z_sigma
        self.name = name

    def output_tensor(self, input_=None):
        emb = self.call_layer_on_input(self.embedding, input_)
        hidden = self.call_layer_on_input(self.hidden, emb)
        z_mean = self.call_layer_on_input(self.z_mean, hidden)
        z_sigma = self.call_layer_on_input(self.z_sigma, hidden)
        z = self.call_layer_on_input(layers.Lambda(self.sampling, output_shape=(self.latent_dim,)), ([z_mean, z_sigma]))

        return z_mean, z_sigma, z

    def __call__(self, input_=None):
        x = input_ if input_ is not None else layers.Input(shape=(self.input_shape[0],))
        z_mean, z_sigma, z = self.output_tensor(x)
        model = models.Model(x, z_mean)
        return model

    def sampling(self, input_: Optional[Any] = None):
        z_mean, z_sigma = input_
        epsilon = backend.random_normal(shape=(self.latent_dim,),
                                        mean=0., stddev=1.)
        return z_mean + backend.exp(z_sigma / 2) * epsilon

    def get_config_dict(self) -> Dict:
        return {
            'type': type(self).__name__,
            'name': self.name,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'input_shape': self.input_shape
        }

    def get_layers_with_names(self) -> Dict[str, layers.Layer]:
        return {

            f'{self.name}_embedding': self.embedding,
            f'{self.name}_hidden': self.hidden,
            f'{self.name}_dense_z_mean': self.z_mean,
            f'{self.name}_dense_z_sigma': self.z_sigma,
        }

    @classmethod
    def from_config_dict_and_layer_collection(
            cls,
            config_dict: Dict,
            layer_collection: model.ModelLayerCollection,
    ) -> "AMPEncoder":
        return cls(
            name=config_dict['name'],
            embedding=layer_collection[config_dict['name'] + '_embedding'],
            hidden=layer_collection[config_dict['name'] + '_hidden'],
            hidden_dim=config_dict['hidden_dim'],
            latent_dim=config_dict['latent_dim'],
            input_shape=config_dict['input_shape'],
            dense_z_mean=layer_collection[config_dict['name'] + '_dense_z_mean'],
            dense_z_sigma=layer_collection[config_dict['name'] + '_dense_z_sigma'],
        )


class AMPEncoderFactory:

    @staticmethod
    def get_default(
            hidden_dim: int,
            latent_dim: int,
            max_length: int,
    ) -> AMPEncoder:
        emb = layers.Embedding(
            input_dim=21,
            output_dim=100,
            input_length=max_length,
            mask_zero=False
        )
        hidden = layers.GRU(
            hidden_dim,
            return_sequences=False,
        )
        dense_z_mean = layers.Dense(latent_dim)
        dense_z_sigma = layers.Dense(latent_dim)
        return AMPEncoder(
            embedding=emb,
            hidden=hidden,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            input_shape=(max_length, 21),
            dense_z_mean=dense_z_mean,
            dense_z_sigma=dense_z_sigma,
        )
