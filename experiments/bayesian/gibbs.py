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
def make_blocks(D: int, infer_H: bool, infer_m0: bool):
    """
    
    Parameters
    ----------
    D: latent state dimension (number of bonds)
    """

    H_block = _construct_H_block(D)
    m0_block = _construct_m0_block(D)

    blocks = []
    if infer_H:
        blocks.append(H_block)
    if infer_m0:
        blocks.append(m0_block)
    return blocks
    # return (H_block, ) #  m0_block, )


def _construct_m0_block(D):

    def _prior(params: dict):
        """
        """
        mean_m0, cov_m0 = params["mean_m0"], params["cov_m0"]
        prec = solve(cov_m0, jnp.eye(D))
        prec_mean = prec @ mean_m0
        return GaussianNatParam(precision=prec, precision_mean=prec_mean)

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


def _construct_H0_block(D):

    def _prior(params: dict):
        alpha = jnp.broadcast_to(jnp.asarray(params["alpha"]), (D,))
        beta = jnp.broadcast_to(jnp.asarray(params["beta"]), (D,))
        return InverseGammaNatParam(alpha=alpha, beta=beta)

    def _likelihood(context: GibbsContext):
        """
        Construct the likelihood p(eta_0 | m_0, H_0) as a InverseGamma function of H_0
        """
        m0 = context.params["m0"]
        eta1 = context.trajectory[1][0]
        return InverseGammaNatParam.from_gaussian(value=eta1, mean=m0)

    def _unpack(H0_diag: Array):
        return {"H0": jnp.diag(H0_diag)}

    return ConjugateBlock(
        name="H0",
        prior=_prior,
        likelihood=_likelihood,
        unpack=_unpack
    )


def _construct_H_block(D):

    def _prior(key: PRNGKey, params: dict):
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
        prior=_prior,
        kernel=_kernel
    )