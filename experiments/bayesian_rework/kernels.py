from enum import Enum

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map, tree_leaves
from jax.scipy.linalg import solve_triangular

import rbsmc.csmc as csmc
import rbsmc.rb_csmc as rb_csmc
from rbsmc.utils.mvn import mvn_logpdf

from experiments.bayesian.prior import (log_p0, log_pt, log_potential, 
                                     unpack_params, ou_diag_transition,
                                     _construct_cov_cholesky)



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
        self.N = N
        self.D = D
        self.dts = dts
        self.params = None

    def M0_rvs(self, key, _):

        chol_Q0 = self.params["chol_Q0"]
        chol_H0 = self.params["chol_H0"]

        D = chol_Q0.shape[-1]
        eps_z, eps_eta = jr.normal(key, shape=(2, self.N+1, D))

        # bootstrap from prior
        z = eps_z @ chol_Q0.T
        eta = eps_eta @ chol_H0.T
        return (z, eta)

    def Mt_rvs(self, key, x_t_m_1, inp):
        """
        Parameters
        ----------
        xp:  (z_t_m_1, eta_t_m_1) where
                - z_t_m_1:   (N, D)
                - eta_t_m_1: (N, D)
        """
        chol_H = self.params["chol_H"]
        
        F_t, chol_Q_t, dt, _ = inp
        z_t_m_1, eta_t_m_1 = x_t_m_1

        D = eta_t_m_1.shape[-1]
        eps_z, eps_eta = jr.normal(key, shape=(2, self.N+1, D))

        # bootstrap from prior
        z_t = z_t_m_1 @ F_t.T + eps_z @ chol_Q_t.T
        eta_t = eta_t_m_1 + jnp.sqrt(dt) * (eps_eta @ chol_H.T)
        return (z_t, eta_t)

    def M0_logpdf(self, x0, inp, constant: bool):
        """ Implement logpdf for t=0 proposal kernel """
        return log_p0(self.params, x0, constant=constant)

    def Mt_logpdf(self, xp, x, inp, constant: bool): 
        """ Implement logpdf for Markov proposal kernel """
        _, _, dt, *_ = inp
        return log_pt(self.params, xp, x, dt, constant=constant)

    def G0_logpdf(self, x0, inp):
        data = inp[-1]
        return log_potential(self.params, x0, data)

    def Gt_logpdf(self, x, inp):
        """ Implement logpdf for potential function """
        data = inp[-1]
        return log_potential(self.params, x, data)

    def init(self, key: PRNGKey, data: tuple[Array], **kwargs):
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
        K = self.dts.shape[0]  # number of time increments
        dummy_x = (jnp.zeros((K, self.D)), jnp.zeros((K, self.D)))
        dummy_state = (dummy_x, jnp.zeros((K,), dtype=int))

        # pass dummy state into an unconditional SMC run (state only used for shape)
        kernel = self.get_kernel(dummy_state, data, conditional=False, **kwargs)
        return kernel(key)

    def get_kernel(self, state, data, conditional: bool, **kwargs):
        """
        
        Parameters
        ----------

        Returns
        -------
        """
        
        # precomputations
        A = self.params["A"]
        chol_Q = self.params["chol_Q"]
        Fs, chol_Qs = vmap(lambda dt: ou_diag_transition(A, chol_Q, dt))(self.dts)

        # define inputs
        inp_0 = (self.dts[0], tree_map(lambda x: x[0], data))
        inps = (Fs[1:], chol_Qs[1:], self.dts[1:], tree_map(lambda x: x[1:], data)) 

        # close over pdfs 
        M0_logpdf = lambda _x: self.M0_logpdf(_x, None, False)
        Mt_logpdf = lambda _xp, _x, _inp: self.Mt_logpdf(_xp, _x, _inp, False)
        Gamma_0 = lambda _x: self.Gamma_0(_x, inp_0)
        Gamma_t = lambda _xp, _x, _inp: self.Gamma_t(_xp, _x, _inp)

        # pack functions
        M0 = self.M0_rvs, M0_logpdf
        Mt = self.Mt_rvs, Mt_logpdf, inps
        Gamma_t_plus_params = Gamma_t, inps

        kernel = lambda _k: csmc.kernel(
            _k, state[0], state[1], 
            M0, Gamma_0, Mt, Gamma_t_plus_params, 
            N=self.N, conditional=conditional,
            **kwargs
        )

        return kernel
