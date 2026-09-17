from enum import Enum

import jax.numpy as jnp
import jax.random as jr

from jax import Array, vmap
from jax.random import PRNGKey
from jax.tree_util import tree_map

import rbsmc.csmc as csmc
import rbsmc.rb_csmc as rb_csmc
import rbsmc.gueant as gueant

from rbsmc.bayesian.smc import FeynmanKac, Reference

from experiments.bayesian.prior import log_p0, log_pt, log_ht, ou_diag_transition


class KernelType(Enum):
    CSMC = 0
    RB_CSMC = 1
    GUEANT = 2

    @property
    def kernel_maker(self):
        if self == KernelType.CSMC:
            return CSMC
        elif self == KernelType.RB_CSMC:
            return RBcSMC
        elif self == KernelType.GUEANT:
            return GueantCSMC
        else:
            raise NotImplementedError


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

    def M0_rvs(self, params, key, N):

        m0 = params["m0"]
        Q0 = params["Q0"]
        H0 = params["H0"]
        chol_Q0 = jnp.linalg.cholesky(Q0)
        chol_H0 = jnp.linalg.cholesky(H0)

        D = chol_Q0.shape[-1]
        eps_z, eps_eta = jr.normal(key, shape=(2, N, D))

        # bootstrap from prior
        z = eps_z @ chol_Q0.T
        eta = m0 + eps_eta @ chol_H0.T
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

        N, D = eta_t_m_1.shape
        eps_z, eps_eta = jr.normal(key, shape=(2, N, D))

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
        state:     Shared ``Reference`` object.
        """
        K = self.dts.shape[0] + 1
        dummy_x = (jnp.zeros((K, self.D)), jnp.zeros((K, self.D)))
        dummy_reference = Reference(dummy_x, jnp.zeros((K,), dtype=int))

        kernel = self.get_kernel(params, dummy_reference, data, conditional=False, **kwargs)
        reference, _ = kernel(key)
        return reference

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
        Gamma_0 = lambda _x: M0_logpdf(_x) + self.G0_logpdf(params, _x, inp_0)
        Gamma_t = lambda _xp, _x, _inp: (
            Mt_logpdf(_xp, _x, _inp) + self.Gt_logpdf(params, _x, _inp)
        )

        # pack functions
        M0 = M0_rvs, M0_logpdf
        Mt = Mt_rvs, Mt_logpdf, inps
        Gamma_t_plus_params = Gamma_t, inps

        kernel = lambda _k: csmc.kernel(
            _k, state,
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
        N:    Number of particles excluding the retained reference particle
        D:    Latent state dimension
        dts:  (K-1,) transition times where K is the number of observations
        """

        self.N = N
        self.D = D
        self.dts = dts

    def M0(self, params, num_particles: int):
        """Return the initial Gaussian state before coordinate conditioning."""
        Q0 = params["Q0"]
        H0 = params["H0"]
        m0 = params["m0"]
        z_m0 = jnp.zeros((num_particles, self.D))
        eta_m0 = jnp.broadcast_to(m0, shape=(num_particles, self.D))
        return rb_csmc.GaussianState((z_m0, eta_m0), (Q0, H0))

    def Mt(self, params, x_t_m_1, P_t_m_1, inp):
        """Return all affine Gaussian prediction quantities for one step."""
        H = params["H"]
        F_t, chol_Q_t, dt, _ = inp
        z_t_m_1, eta_t_m_1 = x_t_m_1
        Q_t_m_1, H_t_m_1 = P_t_m_1

        Q_t = chol_Q_t @ chol_Q_t.T
        Q_pred = F_t @ Q_t_m_1 @ F_t.T + Q_t
        H_pred = H_t_m_1 + dt * H
        m_pred_z = z_t_m_1 @ F_t.T
        m_pred_eta = eta_t_m_1

        I = jnp.eye(self.D)
        means = m_pred_z, m_pred_eta
        covariances = Q_pred, H_pred
        cross_covariances = Q_t_m_1 @ F_t.T, H_t_m_1
        transitions = F_t, I
        offsets = jnp.zeros((self.D,)), jnp.zeros((self.D,))
        return rb_csmc.GaussianPrediction(
            means, covariances, cross_covariances, transitions, offsets
        )

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
        state:     Shared ``Reference`` object. RB coordinates are extracted
                   from its full trajectory when the next kernel is built.
        """
        K = self.dts.shape[0] + 1
        dummy_x = (jnp.zeros((K, self.D)), jnp.zeros((K, self.D)))
        dummy_reference = Reference(dummy_x, jnp.zeros((K,), dtype=int))

        kernel = self.get_kernel(params, dummy_reference, data, conditional=False, **kwargs)
        reference, _ = kernel(key)
        return reference

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

        M_0 = lambda num_particles: self.M0(params, num_particles)
        M_t = lambda means, covariances, inp: self.Mt(
            params, means, covariances, inp
        )
        G_0 = lambda x: self.G0_logpdf(params, x, inp_0)
        G_t = lambda xp, x, inp: self.Gt_logpdf(params, x, inp)

        return lambda key: rb_csmc.kernel(
            key, state, indices,
            M_0, G_0, (M_t, inps), (G_t, inps),
            N=self.N, conditional=conditional,
            **kwargs
        )


class GueantCSMC(FeynmanKac):

    name: str = "GUEANT"

    def __init__(
            self,
            N: int,
            D: int,
            dts: Array,
        ):
        """
        Parameters
        ----------
        N:    Number of particles excluding the retained reference particle
        D:    Latent state dimension
        dts:  (K-1,) transition times where K is the number of observations
        """
        self.N = N
        self.D = D
        self.dts = dts

    def M0_rvs(self, params, key, N, inp):
        """ Sample from the initial full-state distribution """
        Q0 = params["Q0"]
        H0 = params["H0"]
        m0 = params["m0"]

        key_z, key_eta = jr.split(key)
        chol_Q0 = jnp.linalg.cholesky(Q0)
        chol_H0 = jnp.linalg.cholesky(H0)

        z_0 = jr.normal(key_z, shape=(N, self.D)) @ chol_Q0.T
        eta_0 = m0 + jr.normal(key_eta, shape=(N, self.D)) @ chol_H0.T
        return z_0, eta_0

    def Mt_rvs(self, params, key, xp, inp):
        return None

    def M0_logpdf(self, params, x0, inp, constant: bool):
        """ Evaluate the initial full-state density """
        return log_p0(params, x0, constant=constant)

    def Mt_logpdf(self, params, x_t_m_1, x_t, inp, constant: bool):
        """ Evaluate the full-state transition density """
        _, _, dt, _ = inp
        return log_pt(params, x_t_m_1, x_t, dt=dt, constant=constant)

    def G0_logpdf(self, params, x0, inp):
        """ Evaluate the initial emission density """
        data = inp[-1]
        return log_ht(params, x0, data)

    def Gt_logpdf(self, params, x_t, inp):
        """ Evaluate the emission density """
        data = inp[-1]
        return log_ht(params, x_t, data)

    def Gamma_0_logpdf(self, params, x0, inp):
        """ Evaluate the initial unnormalised Feynman-Kac density """
        return self.M0_logpdf(params, x0, inp, constant=True) + self.G0_logpdf(params, x0, inp)

    def Gamma_t_logpdf(self, params, x_t_m_1, x_t, inp):
        """ Evaluate the subsequent unnormalised Feynman-Kac density """
        return self.Mt_logpdf(params, x_t_m_1, x_t, inp, constant=True) + self.Gt_logpdf(params, x_t, inp)

    def init(self, key: PRNGKey, params: dict, data: tuple[Array], **kwargs):
        """
        Initialise the retained trajectory using an unconditional SMC run.
        """
        K = self.dts.shape[0] + 1
        dummy_x = (jnp.zeros((K, self.D)), jnp.zeros((K, self.D)))
        dummy_reference = Reference(dummy_x, jnp.zeros((K,), dtype=int))

        kernel = self.get_kernel(params, dummy_reference, data, conditional=False, **kwargs)
        reference, _ = kernel(key)
        return reference

    def get_kernel(
            self,
            params: dict,
            state,
            data: tuple[Array],
            conditional: bool,
            **kwargs
        ):
        A = params["A"]
        Q = params["Q"]
        H = params["H"]
        R = params["R"]
        alpha = params["alpha"]
        psi = params["psi"]

        chol_H = jnp.linalg.cholesky(H)
        chol_R = jnp.linalg.cholesky(R)
        Fs, chol_Qs = vmap(lambda dt: ou_diag_transition(A, Q, dt))(self.dts)

        inp_0 = tree_map(lambda x: x[0], data),
        inps = Fs, chol_Qs, self.dts, tree_map(lambda x: x[1:], data),

        M_0_rvs = lambda key, N: self.M0_rvs(params, key, N, inp_0)
        M_0_logpdf = lambda x: self.M0_logpdf(params, x, inp_0, constant=True)
        Gamma_0 = lambda x: self.Gamma_0_logpdf(params, x, inp_0)
        Gamma_t = lambda xp, x, inp: self.Gamma_t_logpdf(params, xp, x, inp)

        M_0 = M_0_rvs, M_0_logpdf
        Gamma_t_plus_params = Gamma_t, inps

        return lambda key: gueant.kernel(
            key, state,
            M_0, Gamma_0,
            inps, Gamma_t_plus_params,
            chol_H, chol_R, alpha, psi,
            N=self.N, conditional=conditional,
            **kwargs
        )
