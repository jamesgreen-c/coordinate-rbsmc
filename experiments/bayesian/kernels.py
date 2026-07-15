from enum import Enum

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map, tree_leaves
from jax.scipy.linalg import solve_triangular

import rbsmc.csmc as csmc
import rbsmc.rb_csmc as rb_csmc
from rbsmc.utils.math import mvn_logpdf

from experiments.bayesian.prior import (log_p0, log_pt, log_potential, 
                                     unpack_params, ou_diag_transition,
                                     _construct_cov_cholesky)

def inflate_observed_coord(u_i, i, D):
    return jnp.full(u_i.shape + (D,), jnp.nan, dtype=u_i.dtype).at[..., i].set(u_i)

class KernelType(Enum):
    CSMC = 0
    RB_CSMC = 1

    @property
    def kernel_maker(self):
        if self == KernelType.CSMC:
            return get_csmc_kernel
        elif self == KernelType.RB_CSMC:
            return get_rb_csmc_kernel
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
        chol_H = _construct_cov_cholesky(_params["beta"], _params["delta"])

        # get path shape
        D = chol_Q.shape[-1]
        T = ys.shape[-1]

        # precompute exact OU transition dynamics
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
            F_t, chol_Q_t, dt, _ = params
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
        inps = (Fs[1:], chol_Qs[1:], dts[1:], tree_map(lambda x: x[1:], data)) 
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


def get_rb_csmc_kernel(N, dts, conditional: bool = False, sweeps: int = 1, **kwargs):
    """
    Constructs a bootstrap Rao-Blackwellised CSMC kernel
    
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
        chol_H = _construct_cov_cholesky(_params["beta"], _params["delta"])

        # get path shape
        D = chol_Q.shape[-1]
        T = ys.shape[-1]

        # precompute exact OU transition dynamics and inverse cholesky factors
        H = chol_H @ chol_H.T
        Fs, chol_Qs = jax.vmap(lambda dt: ou_diag_transition(A, chol_Q, dt))(dts)
        inv_chol_Q0 = solve_triangular(chol_Q0, jnp.eye(D), lower=True)
        inv_chol_H0 = solve_triangular(chol_H0, jnp.eye(D), lower=True)

        ###################
        #    filtering    #
        ###################
        def M0_rvs(key, _):
            i = indices[0]
            m0 = jnp.zeros((N+1, D))
            eps_z, eps_eta = jr.normal(key, shape=(2, N+1))
            
            # log half-spread
            P_pred_z = chol_Q0 @ chol_Q0.T 
            z_i = eps_z * jnp.sqrt(P_pred_z[i, i])

            # mid-YtB
            P_pred_eta = chol_H0 @ chol_H0.T 
            eta_i = eps_eta * jnp.sqrt(P_pred_eta[i, i])
            
            u0 = (z_i, eta_i)
            m_pred = (m0, m0)
            P_pred = (P_pred_z, P_pred_eta)
            return u0, m_pred, P_pred
        
        def Mt_rvs(key, x_t_m_1, P_t_m_1, params):
            """
            Parameters
            ----------
            x_t_m_1:  (means_z, means_eta) where, for particles N and dimension D,
                        - means_z:   (N, D)
                        - means_eta: (N, D)
            P_t_m_1:  (P_pred_z, P_pred_eta) where
                        - P_z:    (D, D)
                        - P_eta:  (D, D)
            """
            F_t, chol_Q_t, dt, data_t = params
            i_t = data_t[1]

            z_t_m_1, eta_t_m_1 = x_t_m_1
            P_z, P_eta = P_t_m_1
            eps_z, eps_eta = jr.normal(key, shape=(2, N+1))

            # sample log half-spread at index i
            Q = chol_Q_t @ chol_Q_t.T
            P_pred_z = (F_t @ P_z @ F_t.T) + Q
            m_pred_z = z_t_m_1 @ F_t.T
            z_i = m_pred_z[:, i_t] + eps_z * jnp.sqrt(P_pred_z[i_t, i_t])

            # sample mid-YtB for index id
            P_pred_eta = P_eta + (dt * H)
            m_pred_eta = eta_t_m_1
            eta_i = m_pred_eta[:, i_t] + eps_eta * jnp.sqrt(P_pred_eta[i_t, i_t])

            u_t = (z_i, eta_i)
            m_pred_t = (m_pred_z, m_pred_eta)
            P_pred_t = (P_pred_z, P_pred_eta)
            return u_t, m_pred_t, P_pred_t
        
        def G_0(u):
            i = indices[0]
            z_i, eta_i = u

            # fake it till u make it
            z = inflate_observed_coord(z_i, i, D)
            eta = inflate_observed_coord(eta_i, i, D)
            val = log_potential(prior_params, (z, eta), data_0)
            return val
        
        def G_t(x_t_m_1, u_t, params):
            _, _, _, data_t = params
            z_i, eta_i = u_t
            i = data_t[1]

            # fake it till u make it
            z_t = inflate_observed_coord(z_i, i, D)
            eta_t = inflate_observed_coord(eta_i, i, D)
            val = log_potential(prior_params, (z_t, eta_t), data_t)
            return val
        
        ###################
        #    smoothing    #
        ###################
        def M0_logpdf(x):
            z, eta = x
            m0 = jnp.zeros((N+1, D))
            val = mvn_logpdf(z, m0, None, chol_inv=inv_chol_Q0, constant=False)
            val += mvn_logpdf(eta, m0, None, chol_inv=inv_chol_H0, constant=False)
            return val

        def Mt_logpdf(x_t_m_1, P_t_m_1, x_t, params):
            """
            Log PDF calculated over whole vector x_t rather than single coords u_t = (z_i, eta_i)
            """
            F_t, chol_Q_t, dt, _ = params
            z_t_m_1, eta_t_m_1 = x_t_m_1
            z_t, eta_t = x_t
            P_z, P_eta = P_t_m_1

            # calculate log half-spread logpdf
            m_pred_z = z_t_m_1 @ F_t.T
            Q = chol_Q_t @ chol_Q_t.T
            P_pred_z = (F_t @ P_z @ F_t.T) + Q
            chol_P_pred_z = jnp.linalg.cholesky(P_pred_z)
            inv_chol_P_pred_z = solve_triangular(chol_P_pred_z, jnp.eye(D), lower=True)
            val = mvn_logpdf(z_t, m_pred_z, None, chol_inv=inv_chol_P_pred_z, constant=False)

            # calculate mid-YtB logpdf
            m_pred_eta = eta_t_m_1
            P_pred_eta = P_eta + (dt * H)
            chol_P_pred_eta = jnp.linalg.cholesky(P_pred_eta)
            inv_chol_P_pred_eta = solve_triangular(chol_P_pred_eta, jnp.eye(D), lower=True)
            val += mvn_logpdf(eta_t, m_pred_eta, None, chol_inv=inv_chol_P_pred_eta, constant=False)

            m_pred = (m_pred_z, m_pred_eta)
            P_pred = (P_pred_z, P_pred_eta)
            return val, m_pred, P_pred
        
        def Gamma_t(x_t_m_1, P_t_m_1, x_t, params):
            F_t, *_ = params
            P_z, P_eta = P_t_m_1
            z_t, eta_t = x_t
            z_t_m_1, eta_t_m_1 = x_t_m_1

            val, m_pred, P_pred = Mt_logpdf(x_t_m_1, P_t_m_1, x_t, params)
            # u_t = tree_map(lambda u: u[i_t], x_t)
            # val += G_t(x_t_m_1, u_t, params)
            
            m_pred_z, m_pred_eta = m_pred
            P_pred_z, P_pred_eta = P_pred

            J_z = P_z @ F_t.T @ jnp.linalg.inv(P_pred_z)
            m_smooth_z = z_t_m_1 + (z_t - m_pred_z) @ J_z.T
            P_smooth_z = P_z - J_z @ P_pred_z @ J_z.T
            P_smooth_z = 0.5 * (P_smooth_z + P_smooth_z.T)

            J_eta = P_eta @ jnp.linalg.inv(P_pred_eta)
            m_smooth_eta = eta_t_m_1 + (eta_t - m_pred_eta) @ J_eta.T
            P_smooth_eta = P_eta - J_eta @ P_eta
            P_smooth_eta = 0.5 * (P_smooth_eta + P_smooth_eta.T)

            m_smooth = (m_smooth_z, m_smooth_eta)
            P_smooth = (P_smooth_z, P_smooth_eta)
            return val, m_smooth, P_smooth

        inps = (Fs[1:], chol_Qs[1:], dts[1:], tree_map(lambda x: x[1:], data)) 
        M0 = M0_rvs, M0_logpdf
        Mt = Mt_rvs, Mt_logpdf, inps
        G_t_plus_params = G_t, inps
        Gamma_t_plus_params = Gamma_t, inps

        # get independent smoothing samples
        if conditional:
            apply = lambda _k, _state: rb_csmc.kernel(
                _k, 
                _state[0], _state[1], 
                indices,
                M0, G_0, 
                Mt, G_t_plus_params, Gamma_t_plus_params,
                N=N+1,
                conditional=conditional,
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

        sample = lambda _k: rb_csmc.kernel(_k, 
                                           state[0], state[1], 
                                           indices,
                                           M0, G_0, 
                                           Mt, G_t_plus_params, Gamma_t_plus_params,
                                           N=N+1,
                                           conditional=conditional,
                                           **kwargs)

        return jax.vmap(sample, in_axes=(0))(keys)
    
    init = lambda x: (x, jnp.zeros((tree_leaves(x)[0].shape[0],), dtype=int))
    return kernel, init
