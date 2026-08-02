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



######################################
#       CSMC Feynman-Kac Model       # 
######################################

from jax import Array, vmap
from jax.random import PRNGKey
from rbsmc.bayesian.smc import FeynmanKac

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
        I know aprior the latent state is two components 
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
        return kernel(key)

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
        M0_logpdf = lambda _x: self.M0_logpdf(params, _x, inp_0, False)
        Mt_logpdf = lambda _xp, _x, _inp: self.Mt_logpdf(params, _xp, _x, _inp, False)
        Gamma_0 = lambda _x: self.Gamma_0(params, _x, inp_0)
        Gamma_t = lambda _xp, _x, _inp: self.Gamma_t(params, _xp, _x, _inp)

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
