from enum import Enum
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from jax.random import PRNGKey
from jax import Array

import numpy as np
from jax.scipy.stats import norm
from jax.scipy.linalg import solve_triangular
from jax.tree_util import tree_map

from experiments.corporate_bonds.model import log_potential, ou_diag_transition

from rbsmc.utils.math import mvn_logpdf
from rbsmc.utils.mcmc_utils import aux_sampling_routine, delta_adaptation_routine
from rbsmc import rb_csmc
from rbsmc import csmc
# from cd_ssm import gueant as gueant_csmc


class KernelType(Enum):
    CSMC = 0
    RB_CSMC = 1
    GUEANT = 2
    
    @property
    def kernel_maker(self):
        if self == KernelType.CSMC:
            return get_csmc_kernel
        elif self == KernelType.RB_CSMC:
            return get_rb_csmc_kernel
        elif self == KernelType.GUEANT:
            return get_gueant_csmc_kernel
        else:
            raise NotImplementedError

    
    def shape_delta(self, delta, T):
        if self == KernelType.CSMC:
            return delta
        elif self == KernelType.RB_CSMC:
            return delta
        elif self == KernelType.GUEANT:
            return delta
        else:
            return NotImplementedError("Shape delta not implemented for kernel type")


#######################
# Kernel constructors #
#######################

def get_csmc_kernel(
        ys: Array,
        indices: Array,
        obs_types: Array,
        alpha: Array,
        psi: Array,
        A: Array,
        chol_Q0: Array,
        chol_Q: Array,
        chol_H0: Array,
        chol_H: Array,
        chol_R: Array, 
        N, 
        dts,
        style="bootstrap",
        **kwargs
):
    """
    
    Paramters
    ---------
    ys:         (T,) Observation values,
    indices:    (T,) Index of the relevant dimension at observation t
    obs_types:  (T,) The class of observation ie D2C, D2D, RFQ
    alpha:      (D,) The acceptable width of D2D trades for each bond
    psi:        (D,) Half-spread scale
    A:          (D, D) Diagonal transition matrix for the log half-spreads zs
    chol_Q0:    (D, D) Cholesky of the initial covariance matrix of zs
    chol_Q:     (D, D) Cholesky of the covariance matrix of zs
    chol_H0:    (D, D) Cholesky of the initial covariance matrix of mid-prices etas
    chol_H:     (D, D) Cholesky of the covariance matrix of mid-prices etas
    chol_R:     (D, D) Cholesky factor of the covariance of the ys
    N:          The number of particles, 
    dts         The change in time between each observation. dts[0] = 0 
    """
    T = ys.shape[0]
    D = A.shape[0]
    ts = jnp.cumsum(dts)

    # precompute exact OU transition dynamics and inverse cholesky factors
    H = chol_H @ chol_H.T
    Fs, chol_Qs = jax.vmap(lambda dt: ou_diag_transition(A, chol_Q, dt))(dts)
    inv_chol_Q0 = solve_triangular(chol_Q0, jnp.eye(D), lower=True)
    inv_chol_H0 = solve_triangular(chol_H0, jnp.eye(D), lower=True)

    if style == "bootstrap":

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
            _, _, _, F_t, chol_Q_t, dt = params
            z_t_m_1, eta_t_m_1 = x_t_m_1
            eps_z, eps_eta = jr.normal(key, shape=(2, N+1, D))

            # bootstrap from prior
            z_t = z_t_m_1 @ F_t.T + eps_z @ chol_Q_t.T
            eta_t = eta_t_m_1 + eps_eta @ (dt * chol_H).T
            return (z_t, eta_t)
        
        def M0_logpdf(x):
            z, eta = x
            
            m0 = jnp.zeros((N+1, D))
            val = mvn_logpdf(z, m0, None, chol_inv=inv_chol_Q0, constant=False)
            val += mvn_logpdf(eta, m0, None, chol_inv=inv_chol_H0, constant=False)
            return val

        def Mt_logpdf(x_t_m_1, x_t, params):
            _, _, _, F_t, chol_Q_t, dt = params
            z_t_m_1, eta_t_m_1 = x_t_m_1
            z_t, eta_t = x_t

            # calculate log half-spread logpdf
            inv_chol_Q_t = solve_triangular(chol_Q_t, jnp.eye(D), lower=True)
            val = mvn_logpdf(z_t, z_t_m_1 @ F_t.T, None, chol_inv=inv_chol_Q_t, constant=False)

            # calculate mid-YtB logpdf
            inv_chol_H = solve_triangular(dt * chol_H, jnp.eye(D), lower=True)
            val += mvn_logpdf(eta_t, eta_t_m_1, None, chol_inv=inv_chol_H, constant=False)

            return val
        
        def Gamma_0(x):
            i = indices[0]
            z, eta = x
            
            z_i, eta_i = z[..., i], eta[..., i]
            val = log_potential(i, ys[0], z_i, eta_i, obs_types[0], alpha[i], psi, chol_R)
            val += M0_logpdf(x)

            return val

        def Gamma_t(x_t_m_1, x_t, params):
            y_t, obs_type_t, i_t, *_ = params
            z_t, eta_t = x_t

            z_ti, eta_ti = z_t[..., i_t], eta_t[..., i_t]
            val = log_potential(i_t, y_t, z_ti, eta_ti, obs_type_t, alpha[i_t], psi, chol_R)
            val += Mt_logpdf(x_t_m_1, x_t, params)
            
            return val

    inps = (ys[1:], obs_types[1:], indices[1:], Fs[1:], chol_Qs[1:], dts[1:]) 
    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, Mt_logpdf, inps
    Gamma_t_plus_params = Gamma_t, inps

    init = lambda x: (x, jnp.zeros((T,), dtype=int))
    kernel = lambda key, state, *_: csmc.kernel(
        key, 
        state[0], 
        state[1], 
        M0, 
        Gamma_0, 
        Mt, 
        Gamma_t_plus_params,
        N=N+1, 
        **kwargs
    )

    return kernel, init

def get_rb_csmc_kernel(
        ys: Array,
        indices: Array,
        obs_types: Array,
        alpha: Array,
        psi: Array,
        A: Array,
        chol_Q0: Array,
        chol_Q: Array,
        chol_H0: Array,
        chol_H: Array,
        chol_R: Array, 
        N, 
        dts,
        style="bootstrap",
        **kwargs
):
    """
    
    Paramters
    ---------
    ys:         (T,) Observation values,
    indices:    (T,) Index of the relevant dimension at observation t
    obs_types:  (T,) The class of observation ie D2C, D2D, RFQ
    alpha:      (D,) The acceptable width of D2D trades for each bond
    psi:        (D,) Half-spread scale
    A:          (D, D) Diagonal transition matrix for the log half-spreads zs
    chol_Q0:    (D, D) Cholesky of the initial covariance matrix of zs
    chol_Q:     (D, D) Cholesky of the covariance matrix of zs
    chol_H0:    (D, D) Cholesky of the initial covariance matrix of mid-prices etas
    chol_H:     (D, D) Cholesky of the covariance matrix of mid-prices etas
    chol_R:     (D, D) Cholesky factor of the covariance of the ys
    N:          The number of particles, 
    dts         The change in time between each observation. dts[0] = 0 
    """
    T = ys.shape[0]
    D = A.shape[0]
    ts = jnp.cumsum(dts)

    # precompute exact OU transition dynamics and inverse cholesky factors
    H = chol_H @ chol_H.T
    Fs, chol_Qs = jax.vmap(lambda dt: ou_diag_transition(A, chol_Q, dt))(dts)
    inv_chol_Q0 = solve_triangular(chol_Q0, jnp.eye(D), lower=True)
    inv_chol_H0 = solve_triangular(chol_H0, jnp.eye(D), lower=True)

    if style == "bootstrap":

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
            _, _, i_t, F_t, chol_Q_t, dt = params
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
            val = log_potential(i, ys[0], z_i, eta_i, obs_types[0], alpha[i], psi, chol_R)
            return val
        
        def G_t(x_t_m_1, u_t, params):
            y_t, obs_type_t, i_t, *_ = params
            z_i, eta_i = u_t
            val = log_potential(i_t, y_t, z_i, eta_i, obs_type_t, alpha[i_t], psi, chol_R)
            return val
        
        ###################
        #    smoothing    #
        ###################
        def M0_logpdf(x):
            # need to implement the backward sampling stuff from Adrien here
            # t=0 half log-spreads logpdf
            z, eta = x
            m0 = jnp.zeros((N+1, D))
            val = mvn_logpdf(z, m0, None, chol_inv=inv_chol_Q0)
            val += mvn_logpdf(eta, m0, None, chol_inv=inv_chol_H0)
            return val

        def Mt_logpdf(x_t_m_1, P_t_m_1, x_t, params):
            """
            Log PDF calculated over whole vector x_t rather than single coords u_t = (z_i, eta_i)
            """
            # need to implement the backward sampling stuff from Adrien here
            _, _, i_t, F_t, chol_Q_t, dt = params
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
            _, _, i_t, F_t, chol_Q_t, dt = params
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


    inps = (ys[1:], obs_types[1:], indices[1:], Fs[1:], chol_Qs[1:], dts[1:]) 
    M0 = M0_rvs, M0_logpdf
    Mt = Mt_rvs, Mt_logpdf, inps
    G_t_plus_params = G_t, inps
    Gamma_t_plus_params = Gamma_t, inps

    init = lambda x: (x, jnp.zeros((T,), dtype=int))
    kernel = lambda key, state, *_: rb_csmc.kernel(
        key, 
        state[0], 
        state[1], 
        indices,
        M0, 
        G_0, 
        Mt, 
        G_t_plus_params,
        Gamma_t_plus_params,
        N=N+1, 
        **kwargs
    )

    return kernel, init
        
def get_gueant_csmc_kernel(
        obs, 
        A: Array, 
        psi: Array,
        chol_P0_z: Array,
        chol_P0_eta: Array,
        chol_Q_z: Array, 
        chol_Q_eta: Array, 
        chol_R: Array, 
        N, 
        dts, 
        style="guided", 
        **kwargs
    ):
    return None 

