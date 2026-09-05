import jax.numpy as jnp

from jax import vmap, Array
from jax.random import PRNGKey
from jax.scipy.linalg import solve

from rbsmc.bayesian.gibbs import ConjugateBlock, ConditionalBlock, GibbsContext
from rbsmc.utils.horseshoe import Horseshoe
from rbsmc.bayesian.dists import GaussianNatParam, InverseGammaNatParam

##########################
#     horseshoe prior    #
##########################
def make_blocks(D: int, infer_H: bool, infer_m0: bool, infer_H0: bool):
    """
    
    Parameters
    ----------
    D: latent state dimension (number of bonds)
    """

    H_block = _construct_H_block(D)
    m0_block = _construct_m0_block(D)
    H0_xi_block = _construct_auxiliary_H0_block(D)
    H0_block = _construct_H0_block(D)

    blocks = []
    if infer_H:
        blocks.append(H_block)
    if infer_m0:
        blocks.append(m0_block)
    if infer_H0:
        blocks.extend((H0_xi_block, H0_block))
    return blocks


def _construct_m0_block(D, mean=0.5, variance=1.0):

    # prior specification
    mean = jnp.broadcast_to(jnp.asarray(mean), (D,))
    covariance = variance * jnp.eye(D)
    precision = solve(covariance, jnp.eye(D))
    _prior = GaussianNatParam(precision=precision, precision_mean=precision @ mean)

    def _likelihood(context: GibbsContext):
        """
        Construct the likelihood p(eta_0 | m_0, H_0) as a Gaussian function of m_0.
        """
        H0 = context.params["H0"]
        eta1 = context.trajectory[1][0]     # (zs, etas)
        prec = solve(H0, jnp.eye(D))
        prec_mean = prec @ eta1
        return GaussianNatParam(precision=prec, precision_mean=prec_mean)

    def _unpack(sample: Array):
        return {"m0": sample}

    return ConjugateBlock(
        name="m0",
        prior=_prior,
        likelihood=_likelihood,
        unpack=_unpack
    )


def _construct_auxiliary_H0_block(D, scale=1.0):

    alpha=jnp.full((D,), 0.5)
    beta=jnp.full((D,), 1 / scale**2)
    _prior = InverseGammaNatParam(alpha_plus_one=alpha + 1, beta=beta)

    def _likelihood(context: GibbsContext):
        """
        Construct p(H0 | xi) as an inverse-Gamma function of xi.
        """
        H0_diag = jnp.diag(context.params["H0"])
        alpha = jnp.full((D,), -0.5)
        return InverseGammaNatParam(alpha_plus_one=alpha + 1, beta=1 / H0_diag)

    def _unpack(sample: Array):
        return {"H0_xi": sample}

    return ConjugateBlock(
        name="H0_xi",
        prior=_prior,
        likelihood=_likelihood,
        unpack=_unpack,
    )


def _construct_H0_block(D):

    def _prior(params: dict):
        """
        H0_d | xi_d ~ InvGamma(1 / 2, 1 / xi_d).
        """
        xi = params["H0_xi"]
        alpha = jnp.full((D,), 0.5)
        return InverseGammaNatParam(alpha_plus_one=alpha + 1, beta=1 / xi)

    def _likelihood(context: GibbsContext):
        """
        Construct p(eta_1 | m_0, H_0) as an inverse-Gamma function of the diagonal entries of H_0.
        """
        m0 = context.params["m0"]
        eta1 = context.trajectory[1][0]
        return InverseGammaNatParam.from_gaussian(value=eta1, mean=m0)

    def _unpack(sample: Array):
        return {"H0": jnp.diag(sample)}

    return ConjugateBlock(
        name="H0",
        prior=_prior,
        likelihood=_likelihood,
        unpack=_unpack,
    )


# def _construct_H0_block(D, concentration=1.0, scale=0.1):
#     # TODO change to be 2 separate blocks using half-cauchy auxiliary prior

#     alpha = jnp.full((D,), concentration)
#     beta = jnp.full((D,), scale)
#     _prior = InverseGammaNatParam(alpha=alpha, beta=beta)

#     def _likelihood(context: GibbsContext):
#         """
#         Construct the likelihood p(eta_0 | m_0, H_0) as a InverseGamma function of H_0
#         """
#         m0 = context.params["m0"]
#         eta1 = context.trajectory[1][0]
#         return InverseGammaNatParam.from_gaussian(value=eta1, mean=m0)

#     def _unpack(H0_diag: Array):
#         return {"H0": jnp.diag(H0_diag)}

#     return ConjugateBlock(
#         name="H0",
#         prior=_prior,
#         likelihood=_likelihood,
#         unpack=_unpack
#     )


def _construct_H_block(D):

    def _initialiser(key: PRNGKey):
        H, beta, llambda, nu, tau, xi = Horseshoe.init(D)
        return {"H": H, "beta": beta, "llambda": llambda, "nu": nu, "tau": tau, "xi": xi}

    def _kernel(key: PRNGKey, context: GibbsContext):
        _params = context.params
        _, _etas = context.trajectory    # (K, D)
        dts = context.dts                # (K-1,)

        # form required quantities
        K = _etas.shape[0]                                       # number of observations
        increments = _etas[1:, :] - _etas[:-1, :]                # (K-1, D)
        residuals = increments / jnp.sqrt(dts[:, None])          # (K-1, D)
        scatter = jnp.einsum("td,te->de", residuals, residuals)  # (D, D)

        # scatter += 0.05 * jnp.eye(D)                             # small ridge 

        # sample
        H, beta, llambda, nu, tau, xi = Horseshoe.sample(
            key=key, 
            N=K-1,
            scatter=scatter,
            beta=_params["beta"],
            Q=_params["H"],
            llambda=_params["llambda"],
            nu=_params["nu"],
            tau=_params["tau"],
            xi=_params["xi"]
        )
        return {"H": H, "beta": beta, "llambda": llambda, "nu": nu, "tau": tau, "xi": xi}
    

    return ConditionalBlock(
        names=("H", "beta", "llambda", "nu", "tau", "xi",),
        initialiser=_initialiser,
        kernel=_kernel
    )