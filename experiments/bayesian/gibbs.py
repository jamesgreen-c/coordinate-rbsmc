import jax.numpy as jnp

from jax import vmap, Array
from jax.random import PRNGKey

from rbsmc.bayesian.gibbs import ConjugateBlock, ConditionalBlock, GibbsContext
from rbsmc.utils.horseshoe import Horseshoe


##########################
#     horseshoe prior    #
##########################
def make_blocks(D: int):
    """
    
    Parameters
    ----------
    D: latent state dimension (number of bonds)
    """

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
    

    H_prior = ConditionalBlock(
        names=("H", "beta", "llambda", "nu", "tau", "xi",),
        prior=_prior,
        kernel=_kernel
    )

    return (H_prior, )