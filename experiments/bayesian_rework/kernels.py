from enum import Enum

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map, tree_leaves
from jax.scipy.linalg import solve_triangular

import rbsmc.csmc as csmc
import rbsmc.rb_csmc as rb_csmc
from rbsmc.utils.mvn import mvn_logpdf

from experiments.bayesian_rework.prior import log_p0, log_pt, log_ht, ou_diag_transition

from jax import Array, vmap
from jax.random import PRNGKey
from rbsmc.bayesian.smc import FeynmanKac

######################################
#       CSMC Feynman-Kac Model       # 
######################################

class CSMC(FeynmanKac):

    name: str = "CSMC"

    def __init__(
            self, 
            N: int,
            D: int,
            dts: Array, 
        ):
        """
        Parameters
        ----------
        N:    Number of particles
        D:    Latent state dimension
        dts:  (K-1, D) transition times has K-1 length where K is number of observations
        """

        self.N = N
        self.D = D
        self.dts = dts

    def M0_rvs(self, params, key, _):

        Q0 = params["Q0"]
        H0 = params["H0"]
        chol_Q0 = jnp.linalg.cholesky(Q0)
        chol_H0 = jnp.linalg.cholesky(H0)

        D = chol_Q0.shape[-1]
        eps_z, eps_eta = jr.normal(key, shape=(2, self.N+1, D))

        # bootstrap from prior
        z = eps_z @ chol_Q0.T
        eta = eps_eta @ chol_H0.T
        return (z, eta)

    def Mt_rvs(self, params, key, x_t_m_1, inp):
        """
        Parameters
        ----------
        xp:  (z_t_m_1, eta_t_m_1) where
                - z_t_m_1:   (N, D)
                - eta_t_m_1: (N, D)
        """
        H = params["H"]
        chol_H = jnp.linalg.cholesky(H)
        
        F_t, chol_Q_t, dt, _ = inp
        z_t_m_1, eta_t_m_1 = x_t_m_1

        D = eta_t_m_1.shape[-1]
        eps_z, eps_eta = jr.normal(key, shape=(2, self.N+1, D))

        # bootstrap from prior
        z_t = z_t_m_1 @ F_t.T + eps_z @ chol_Q_t.T
        eta_t = eta_t_m_1 + jnp.sqrt(dt) * (eps_eta @ chol_H.T)
        return (z_t, eta_t)

    def M0_logpdf(self, params, x0, inp, constant: bool):
        """ Implement logpdf for t=0 proposal kernel """
        return log_p0(params, x0, constant=constant)

    def Mt_logpdf(self, params, xp, x, inp, constant: bool): 
        """ Implement logpdf for Markov proposal kernel """
        _, _, dt, *_ = inp
        return log_pt(params, xp, x, dt, constant=constant)

    def G0_logpdf(self, params, x0, inp):
        data = inp[-1]
        return log_ht(params, x0, data)

    def Gt_logpdf(self, params, x, inp):
        """ Implement logpdf for potential function """
        data = inp[-1]
        return log_ht(params, x, data)

    def init(self, key: PRNGKey, params: dict, data: tuple[Array], **kwargs):
        """
        I know apriori the latent state is two components 
            x = (z, eta)
        where z and eta have the same dimensions.

        Parameters 
        ----------
        key:       RNG
        data:      Tuple containing all observation modalities (ie bond idx, trade price etc)
        **kwargs:  Extra keywords for specific kernel being used

        Returns
        -------
        state:     (xs, Bs) for states and (backward) ancestors
        """
        # init a dummy state
        K = self.dts.shape[0] + 1  # number of time increments
        dummy_x = (jnp.zeros((K, self.D)), jnp.zeros((K, self.D)))
        dummy_state = (dummy_x, jnp.zeros((K,), dtype=int))

        # pass dummy state into an unconditional SMC run (state only used for shape)
        kernel = self.get_kernel(params, dummy_state, data, conditional=False, **kwargs)
        xs, Bs, log_ws = kernel(key)
        return (xs, Bs)

    def get_kernel(
            self, 
            params: dict, 
            state: Array, 
            data: tuple[Array], 
            conditional: bool, 
            **kwargs
        ):
        """
        
        Parameters
        ----------

        Returns
        -------
        """
        
        # precomputations
        A = params["A"]
        Q = params["Q"]
        Fs, chol_Qs = vmap(lambda dt: ou_diag_transition(A, Q, dt))(self.dts)  # (K-1, ...)

        # define inputs
        inp_0 = (tree_map(lambda x: x[0], data), )
        inps = (Fs, chol_Qs, self.dts, tree_map(lambda x: x[1:], data)) 

        # close over pdfs 
        M0_rvs = lambda _k, _: self.M0_rvs(params, _k, _)
        Mt_rvs = lambda _k, _xp, _inp: self.Mt_rvs(params, _k, _xp, _inp)
        M0_logpdf = lambda _x: self.M0_logpdf(params, _x, inp_0, constant=True)
        Mt_logpdf = lambda _xp, _x, _inp: self.Mt_logpdf(params, _xp, _x, _inp, constant=True)
        Gamma_0 = lambda _x: self.Gamma_0(params, _x, inp_0, constant=True)
        Gamma_t = lambda _xp, _x, _inp: self.Gamma_t(params, _xp, _x, _inp, constant=True)

        # pack functions
        M0 = M0_rvs, M0_logpdf
        Mt = Mt_rvs, Mt_logpdf, inps
        Gamma_t_plus_params = Gamma_t, inps

        kernel = lambda _k: csmc.kernel(
            _k, state[0], state[1], 
            M0, Gamma_0, Mt, Gamma_t_plus_params, 
            N=self.N, conditional=conditional,
            **kwargs
        )

        return kernel



###################################################
#       Rao-Blackwellised Feynman-Kac Model       #
###################################################

class RBcSMC(FeynmanKac):

    name: str = "RB_CSMC"

    def __init__(
            self, 
            N: int,
            D: int,
            dts: Array, 
        ):
        """
        Parameters
        ----------
        N:    Number of particles
        D:    Latent state dimension
        dts:  (K-1, D) transition times has K-1 length where K is number of observations
        """

        self.N = N
        self.D = D
        self.dts = dts

    def M0_rvs(self, params, key, _, inp):
        """ Only propose particles for the observed coordinate """

        Q0 = params["Q0"]
        H0 = params["H0"]

        data_0 = inp[-1]
        i = data_0[1]

        D = Q0.shape[-1]
        m0 = jnp.zeros((self.N+1, D))

        eps_z, eps_eta = jr.normal(key, shape=(2, self.N+1))
        z_i = eps_z * jnp.sqrt(Q0[i, i])
        eta_i = eps_eta * jnp.sqrt(H0[i, i])

        return (z_i, eta_i), (m0, m0), (Q0, H0)

    def Mt_rvs(self, params, key, x_t_m_1, P_t_m_1, inp):
        """
        Parameters
        ----------
        xp:  (z_t_m_1, eta_t_m_1) where
                - z_t_m_1:   (N, D)
                - eta_t_m_1: (N, D)
        """
        H = params["H"]
        F_t, chol_Q_t, dt, data_t = inp
        i_t = data_t[1]

        z_t_m_1, eta_t_m_1 = x_t_m_1
        Q_t_m_1, H_t_m_1 = P_t_m_1
        eps_z, eps_eta = jr.normal(key, shape=(2, self.N+1))

        # sample log half-spread at index i_t
        Q = chol_Q_t @ chol_Q_t.T
        Q_pred = (F_t @ Q_t_m_1 @ F_t.T) + Q                               # filter covariance
        m_pred_z = z_t_m_1 @ F_t.T                                         # predictive mean over all bond spreads
        z_i = m_pred_z[:, i_t] + eps_z * jnp.sqrt(Q_pred[i_t, i_t])

        # sample mid YtB for index i_t
        H_pred = H_t_m_1 + (dt * H)                                        # filter covariance
        m_pred_eta = eta_t_m_1                                             # predictive mean over all mid prices
        eta_i = m_pred_eta[:, i_t] + eps_eta * jnp.sqrt(H_pred[i_t, i_t])

        u_t = (z_i, eta_i)
        m_pred_t = (m_pred_z, m_pred_eta)
        P_pred_t = (Q_pred, H_pred)
        return u_t, m_pred_t, P_pred_t

    def M0_logpdf(self, params, x0, inp, constant: bool):
        """ Implement logpdf for t=0 proposal kernel """
        return log_p0(params, x0, constant=constant)

    def Mt_logpdf(self, params, x_t_m_1, P_t_m_1, x_t, inp, constant: bool): 
        """ Implement logpdf for Markov proposal kernel """
        H = params["H"]
        F_t, chol_Q_t, dt, _ = inp
        z_t_m_1, eta_t_m_1 = x_t_m_1
        z_t, eta_t = x_t
        Q_t_m_1, H_t_m_1 = P_t_m_1

        D = z_t_m_1.shape[-1]

        # calculate log half-spread logpdf
        Q = chol_Q_t @ chol_Q_t.T
        Q_pred = (F_t @ Q_t_m_1  @ F_t.T) + Q
        chol_Q_pred = jnp.linalg.cholesky(Q_pred)
        inv_chol_Q_pred = solve_triangular(chol_Q_pred, jnp.eye(D), lower=True)
        m_pred_z = z_t_m_1 @ F_t.T
        val = mvn_logpdf(z_t, m_pred_z, None, chol_inv=inv_chol_Q_pred, constant=True)

        # calculate mid-YtB logpdf
        H_pred = H_t_m_1 + (dt * H)
        chol_H_pred = jnp.linalg.cholesky(H_pred)
        inv_chol_H_pred = solve_triangular(chol_H_pred, jnp.eye(D), lower=True)
        m_pred_eta = eta_t_m_1
        val += mvn_logpdf(eta_t, m_pred_eta, None, chol_inv=inv_chol_H_pred, constant=True)

        return val

    def G0_logpdf(self, params, x0, inp):
        data = inp[-1]
        return log_ht(params, x0, data)

    def Gt_logpdf(self, params, x, inp):
        """ Implement logpdf for potential function """
        data = inp[-1]
        return log_ht(params, x, data)

    def rts(self, params, x_t_m_1, P_t_m_1, x_t, inp):
        """
        Calculate p(x_{t-1} | x_t, u_{0:t-1}) for each particle.

        Parameters
        ----------
        x_t_m_1:  Tuple of filtered means at time t-1.
        P_t_m_1:  Tuple of filtered covariances at time t-1.
        x_t:      Sampled full state at time t.
        inp:      Transition inputs for time t.

        Returns
        -------
        m_smooth: Tuple of particle-dependent smoothing means.
        P_smooth: Tuple of particle-independent smoothing covariances.
        """
        H = params["H"]
        F_t, chol_Q_t, dt, _ = inp

        z_t, eta_t = x_t
        z_t_m_1, eta_t_m_1 = x_t_m_1
        Q_t_m_1, H_t_m_1 = P_t_m_1

        # log half-spread RTS update
        Q_t = chol_Q_t @ chol_Q_t.T
        Q_pred = F_t @ Q_t_m_1 @ F_t.T + Q_t
        m_pred_z = z_t_m_1 @ F_t.T

        J_z = Q_t_m_1 @ F_t.T @ jnp.linalg.inv(Q_pred)
        m_smooth_z = z_t_m_1 + (z_t - m_pred_z) @ J_z.T
        Q_smooth = Q_t_m_1 - J_z @ Q_pred @ J_z.T

        # mid-YtB RTS update
        H_pred = H_t_m_1 + dt * H
        m_pred_eta = eta_t_m_1

        J_eta = H_t_m_1 @ jnp.linalg.inv(H_pred)
        m_smooth_eta = eta_t_m_1 + (eta_t - m_pred_eta) @ J_eta.T
        H_smooth = H_t_m_1 - J_eta @ H_pred @ J_eta.T

        # enforce symmetry
        Q_smooth = 0.5 * (Q_smooth + Q_smooth.T)
        H_smooth = 0.5 * (H_smooth + H_smooth.T)

        m_smooth = (m_smooth_z, m_smooth_eta)
        P_smooth = (Q_smooth, H_smooth)
        return m_smooth, P_smooth

    def init(self, key: PRNGKey, params: dict, data: tuple[Array], **kwargs):
        """
        I know apriori the latent state is two components 
            x = (z, eta)
        where z and eta have the same dimensions.

        Parameters 
        ----------
        key:       RNG
        data:      Tuple containing all observation modalities (ie bond idx, trade price etc)
        **kwargs:  Extra keywords for specific kernel being used

        Returns
        -------
        state:     (xs, Bs) for states and (backward) ancestors
        """
        # init a dummy state
        K = self.dts.shape[0] + 1  # number of time increments
        dummy_x = (jnp.zeros((K, self.D)), jnp.zeros((K, self.D)))
        dummy_state = (dummy_x, jnp.zeros((K,), dtype=int))

        # pass dummy state into an unconditional SMC run (state only used for shape)
        kernel = self.get_kernel(params, dummy_state, data, conditional=False, **kwargs)
        xs, Bs, log_ws = kernel(key)
        return (xs, Bs)

    def get_kernel(
            self,
            params: dict,
            state: Array,
            data: tuple[Array],
            conditional: bool,
            **kwargs
        ):
        A = params["A"]
        Q = params["Q"]
        Fs, chol_Qs = vmap(lambda dt: ou_diag_transition(A, Q, dt))(self.dts)

        inp_0 = tree_map(lambda x: x[0], data),
        inps = Fs, chol_Qs, self.dts, tree_map(lambda x: x[1:], data)
        indices = data[1]

        M_0_rvs = lambda key, N: self.M0_rvs(params, key, N, inp_0)
        G_0 = lambda x: self.G0_logpdf(params, x, inp_0)

        M_t_rvs = lambda key, xp, Pp, inp: self.Mt_rvs(params, key, xp, Pp, inp)
        G_t = lambda xp, x, inp: self.Gt_logpdf(params, x, inp)
        M_t_logpdf = lambda xp, Pp, x, inp: self.Mt_logpdf(params, xp, Pp, x, inp, constant=True)
        rts_func = lambda xp, Pp, x, inp: self.rts(params, xp, Pp, x, inp)

        return lambda key: rb_csmc.kernel(
            key, state[0], state[1],
            indices,
            M_0_rvs, G_0, M_t_rvs, G_t,
            M_t_logpdf, rts_func,
            inps,
            N=self.N, conditional=conditional,
            **kwargs
        )