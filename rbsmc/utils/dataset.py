import jax.numpy as jnp

from dataclasses import dataclass
from jax import Array


@dataclass
class Dataset:
    data: tuple[Array, ...]
    states: Array
    params: dict[str, Array]

    @property
    def standardised_data(self):
        means = tuple(jnp.mean(d, axis=(0,1), keepdims=True) for d in self.data)
        stds = tuple(jnp.std(d, axis=(0,1), keepdims=True) for d in self.data)
        scaled_data = tuple((d-m)/s for d, m, s in zip(self.data, means, stds))

        return Dataset(
            data=scaled_data,
            states=self.states,
            params=self.params
        )

    @property
    def flatten(self):
        shape = self.data[0].shape[:2] + (-1,)
        data = tuple(jnp.reshape(x, shape) for x in self.data)

        return Dataset(
            data=data,
            states=self.states,
            params=self.params
        )

    def __getitem__(self, index):
        """Allow indexing over all J data modalities"""
        return Dataset(
            data=tuple(x[index] for x in self.data),
            states=self.states[index],
            params=self.params
        )
        
