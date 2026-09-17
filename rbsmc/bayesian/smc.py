from abc import ABC, abstractmethod
from typing import Any, NamedTuple

from jax import Array
from jax import numpy as jnp
from jax.random import PRNGKey
from jax.tree_util import tree_leaves, tree_structure


class Reference(NamedTuple):
    """A retained trajectory and its particle positions in the last sweep.

    ``ancestors[t]`` is both the particle selected at time ``t`` and the slot
    at which ``trajectory[t]`` is embedded during the next conditional sweep.
    Keeping those positions is what makes ancestor comparison a valid path
    replacement diagnostic for conditional SMC.
    """

    trajectory: Any
    ancestors: Array

    @property
    def particle_indices(self):
        """Backward-compatible alias for older analysis code."""
        return self.ancestors


def _trajectory_replaced(old: Any, new: Any):
    """Compare path values only for genuinely independent SMC sweeps."""
    if tree_structure(old) != tree_structure(new):
        raise ValueError("Old and new trajectories must have the same PyTree structure.")

    replaced = None
    for old_leaf, new_leaf in zip(tree_leaves(old), tree_leaves(new)):
        changed = jnp.not_equal(old_leaf, new_leaf)
        if changed.ndim > 1:
            changed = jnp.any(changed, axis=tuple(range(1, changed.ndim)))
        replaced = changed if replaced is None else jnp.logical_or(replaced, changed)

    if replaced is None:
        raise ValueError("A trajectory must contain at least one array leaf.")
    return replaced


class FeynmanKac(ABC):

    def __init__(self):
        pass

    def M0_rvs(self, params, key: PRNGKey, inp: tuple):
        """Optional initial proposal hook used by full-state kernels."""
        raise NotImplementedError

    def Mt_rvs(self, params, key: PRNGKey, xp, inp: tuple):
        """Optional transition proposal hook used by full-state kernels."""
        raise NotImplementedError

    def M0_logpdf(self, params, x0, inp: tuple, constant: bool):
        """
        Implement logpdf for t=0 proposal kernel

        params:    Dictionary of params required for Markov kernel evaluation.
        x0:        t=0 state vector
        inp:       Any external inputs required for logpdf evaluation e.g. ys[0]
        constant:  Whether the calculate the normalising constant for the logpdf
        """
        raise NotImplementedError

    def Mt_logpdf(self, params, xp, x, inp: tuple, constant: bool): 
        """
        Implement logpdf for Markov proposal kernel

        params:    Dictionary of params required for Markov kernel evaluation.
        xp:        Previous state vector
        x:         Current state vector
        inp:       Any external inputs required for logpdf evaluation e.g. ys[t]
        constant:  Whether the calculate the normalising constant for the logpdf
        """
        raise NotImplementedError

    def G0_logpdf(self, params, x0, inp: tuple):
        raise NotImplementedError

    def Gt_logpdf(self, params, x, inp: tuple):
        """ Implement logpdf for potential function """
        raise NotImplementedError

    def Gamma_0(self, params, x0, inp, constant: bool):
        return self.G0_logpdf(params, x0, inp) + self.M0_logpdf(params, x0, inp, constant=constant)

    def Gamma_t(self, params, xp, x, inp, constant: bool):
        return self.Gt_logpdf(params, x, inp) + self.Mt_logpdf(params, xp, x, inp, constant=constant)

    @abstractmethod
    def init(self, key: PRNGKey, params: dict, data: tuple[Array], **kwargs):
        """ 
        Write a function to get first state for SMC. 
        Usually unconditional run 
        """
        pass

    @abstractmethod
    def get_kernel(self, params, state, data, conditional: bool, **kwargs):
        """ Write a kernel constructor using defined FK methods """
        pass


class SMC(ABC):

    def __init__(self, fk: FeynmanKac, conditional: bool, kwargs: dict):
        self.fk = fk
        self.conditional = conditional
        self.kwargs = kwargs

    def init(self, key: PRNGKey, params: dict, data: tuple[Array]):
        return self.fk.init(key, params, data, **self.kwargs)

    def sample(
            self, 
            key: PRNGKey, 
            params: dict,
            state: Reference,
            data: tuple[Array],
        ):

        # construct new kernel with params
        kernel = self.fk.get_kernel(
            params,
            state, 
            data, 
            conditional=self.conditional, 
            **self.kwargs
        )

        # sample new smoothing path
        reference, log_ws = kernel(key)

        if self.conditional:
            # compare ancestor index
            replaced = reference.ancestors != state.ancestors
        else:
            # labels from independent particle systems are not comparable.
            replaced = _trajectory_replaced(state.trajectory, reference.trajectory)
        return reference, {"replaced": replaced, "log_ws": log_ws}
         

        
