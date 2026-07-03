from enum import Enum
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map, tree_leaves
import numpy as np
from jax.scipy.stats import norm

import rbsmc.csmc as csmc
import rbsmc.rb_csmc as rb_csmc

from experiments.horseshoe.prior import log_p0, log_pt, log_potential, unpack_params, ou_diag_transition


class KernelType(Enum):
    CSMC = 0
    RB_CSMC = 1

    @property
    def kernel_maker(self):
        if self == KernelType.CSMC:
            return get_csmc_kernel
        # elif self == KernelType.RB_CSMC:
        #     return get_rb_csmc_kernel
        else:
            raise NotImplementedError

    def shape_delta(self, delta, T):
        if self == KernelType.CSMC:
            return delta
        elif self == KernelType.RB_CSMC:
            return delta

    def initialise_delta(self, D, T):
        if self == KernelType.CSMC:
            return D ** -1.0
        elif self == KernelType.RB_CSMC:
            return D ** -1.0
        else:
            raise NotImplementedError(f"No delta initialisation for {self}")

#######################
# Kernel constructors #
#######################

def get_csmc_kernel(N, dts, conditional: bool = False, sweeps: int = 1, **kwargs):
    """
    Constructs a bootstrap csmc kernel
    
    """
    kwargs.pop("style")
    sweeps = sweeps if conditional else 1

    def kernel(keys, state, prior_params, data):
        """
        
        Parameters
        ----------
        keys:          (n_samples, ) RNG
        state:         The initial state to start from
        prior_params:  Parameters required for prior chain sampling and evaluation
        means:         (T, D) Means outputted from recognition network for this timeseries obs
        chol_Rs:       (T, D, D) Cholesky of the covariance outputted by the recognition network for this timeseries obs
        all_factors:   [(B, T, D), (B, T, D, D)] all factors outputted by the recognition network to calculate F_{phi}

        Returns
        -------
        samples: (n_samples, T, D)
        """
        ys, indices, obs_types = data
        data_0 = tree_map(lambda x: x[0], data)

        # unpack parameters required for proposals
        _params = unpack_params(prior_params)
        A = _params["A"]
        chol_Q0 = _params["chol_Q0"]
        chol_H0 = _params["chol_H0"]
        chol_Q = _params["chol_Q"]
        chol_H = _params["chol_H"]

        # get path shape
        D = chol_Q.shape[-1]
        T = ys.shape[-1]

        # precompute exact OU transition dynamics and inverse cholesky factors
        Fs, chol_Qs = jax.vmap(lambda dt: ou_diag_transition(A, chol_Q, dt))(dts)

        # Define the Feynman-Kac model
        def M0_rvs(key, _):
            eps_z, eps_eta = jr.normal(key, shape=(2, N+1, D))
            
            # bootstrap from prior
            z = eps_z @ chol_Q0.T
            eta = eps_eta @ chol_H0.T
            return (z, eta) 

        def Mt_rvs(key, x_t_m_1, params):
            """
            Parameters
            ----------
            x_t_m_1:  (z_t, eta_t) where
                        - z_t:   (N, D)
                        - eta_t: (N, D)
            """
            _, _, _, F_t, chol_Q_t, dt, _ = params
            z_t_m_1, eta_t_m_1 = x_t_m_1
            eps_z, eps_eta = jr.normal(key, shape=(2, N+1, D))

            # bootstrap from prior
            z_t = z_t_m_1 @ F_t.T + eps_z @ chol_Q_t.T
            eta_t = eta_t_m_1 + jnp.sqrt(dt) * (eps_eta @ chol_H.T)
            return (z_t, eta_t)

        M0_logpdf = lambda x: log_p0(prior_params, x, constant=False)
        Mt_logpdf = lambda x_t_m_1, x_t, _p: log_pt(prior_params, x_t_m_1, x_t, _p[-2], constant=False)
        Gamma_0 = lambda x: log_potential(prior_params, x, data_0) + M0_logpdf(x)
        Gamma_t = lambda x_t_m_1, x_t, _p: log_potential(prior_params, x_t, _p[-1]) + Mt_logpdf(x_t_m_1, x_t, _p)

        # packing
        inps = (ys[1:], obs_types[1:], indices[1:], Fs[1:], chol_Qs[1:], dts[1:], tree_map(lambda x: x[1:], data)) 
        M0 = M0_rvs, M0_logpdf
        Mt = Mt_rvs, Mt_logpdf, inps
        Gamma_t_plus_params = Gamma_t, inps

        # get independent smoothing samples
        if conditional:
            apply = lambda _k, _state: csmc.kernel(
                _k, _state[0], _state[1], 
                M0, 
                Gamma_0, Mt, Gamma_t_plus_params, 
                N=N, conditional=conditional, 
                **kwargs
            )

            def _body(carry, _k):
                xs, bs = carry
                next_xs, next_bs, _ = apply(_k, (xs, bs))
                return (next_xs, next_bs), next_xs
            
            def scan(_ks):
                samples, _ = jax.lax.scan(_body, carry0, _ks)
                return samples
            
            carry0 = state
            all_keys = jax.vmap(lambda _k: jax.random.split(_k, sweeps))(keys)
            return jax.vmap(scan)(all_keys)

        sample = lambda _k: csmc.kernel(_k, 
                                        state[0], state[1], 
                                        M0, 
                                        Gamma_0, Mt, Gamma_t_plus_params, 
                                        N=N, conditional=conditional, 
                                        **kwargs)

        return jax.vmap(sample, in_axes=(0))(keys)
    
    init = lambda x: (x, jnp.zeros((tree_leaves(x)[0].shape[0],), dtype=int))
    return kernel, init