import jax.random as jr
import blackjax


# MCMC updaters
mwg_init_theta = blackjax.rmh.init
mwg_step_fn_theta = blackjax.rmh.build_kernel()


def mwg_kernel(
        rng_key,
        csmc_kernel,
        param_kernel,
        log_pdf_fn,
        state, 
        parameters
    ):
    """
    Generalised Metropolis-within-Gibbs kernel.

    Parameters
    ----------
    rng_key
        The PRNG key.

    csmc_kernel
        Function implementing the CSMC kernel for updating the latent states.
    param_kernel
        Function implementing the MCMC kernel for updating the parameters.
    log_pdf_fn
        Function computing the joint log-density of the latent states and observations.
    state
        Dictionary with elements `x` and `y`, where the former is an ``RMCState`` object
        and the latter is an ``HMCState`` object.
    parameters
        Dictionary with elements `x` and `y`, each of which is a dictionary of the parameters
        to the corresponding algorithm's ``step_fn()``.

    Returns
    -------
    Dictionary containing the updated ``state``.
    """

    rng_key_x, rng_key_y = jr.split(rng_key, num=2)

    # avoid modifying argument state as JAX functions should be pure
    state = state.copy()

    # --- CSMC update for x ---
    state["x"] = csmc_kernel(rng_key_x, state["x"])

    # --- MWG update for theta ---
    # conditional logdensity of y given x
    def logdensity_theta(theta): return log_pdf_fn(theta=theta, x=state["x"])

    # give state["theta"] the right log_density
    state["theta"] = mwg_init_theta(
        position=state["theta"].position,
        logdensity_fn=logdensity_theta
    )
    # update state["theta"]
    state["theta"], _ = mwg_step_fn_theta(
        rng_key=rng_key_y,
        state=state["theta"],
        logdensity_fn=logdensity_theta,
        **parameters["theta"]
    )

    return state