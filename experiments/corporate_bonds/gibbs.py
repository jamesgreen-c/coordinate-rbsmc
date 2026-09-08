import jax.numpy as jnp
import jax.random as jr
import jax

from jax import vmap, Array
from jax.random import PRNGKey
from jax.scipy.linalg import solve

from rbsmc.utils.horseshoe import Horseshoe
from rbsmc.bayesian.gibbs import ConjugateBlock, ConditionalBlock, GibbsContext
from rbsmc.bayesian.dists import GaussianNatParam, InverseGammaNatParam

##########################
#     horseshoe prior    #
##########################
def make_blocks(D: int, full_inference: bool = False):
    """
    
    Parameters
    ----------
    D: latent state dimension (number of bonds)
    """

    H_block = _construct_H_block(D)
    m0_block = _construct_m0_block(D)
    H0_xi_block = _construct_auxiliary_H0_block(D)
    H0_block = _construct_H0_block(D)
    R_block = _construct_R_block(D)

    blocks = [H_block, m0_block, H0_xi_block, H0_block, R_block]

    if full_inference:
        # TODO: add optionality for inference on PSI, ALPHA, Q, Q0, R as well
        A_block = None
        Q_block = None
        Q0_block = _construct_Q0_block(D)
        PSI_block = None
        ALPHA_block = None
        blocks.extend([A_block, Q_block, Q0_block, PSI_block, ALPHA_block])
        pass 
    
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


def _construct_R_block(D, concentration: float = 1.0, scale: float = 1.0):

    concentration = jnp.full((D,), concentration)
    scale = jnp.full((D,), scale)
    prior = InverseGammaNatParam(alpha_plus_one=concentration + 1, beta=scale)

    def _initialiser(key: PRNGKey):
        variances = prior.dist_param.sample(key, ())
        return {"R": jnp.diag(variances)}

    def _kernel(key: PRNGKey, context: GibbsContext):
        obs_values, bond_idxs, event_types = context.data
        zs, etas = context.trajectory

        PSI = context.params["psi"]
        alpha = context.params["alpha"]
        variances = jnp.diag(context.params["R"])

        auxiliary_key, variance_key = jr.split(key)
        auxiliary_keys = jr.split(auxiliary_key, obs_values.shape[0])

        def _sample_auxiliary(key, y, event, i, z, eta):
            half_spread = PSI[i] * jnp.exp(z[i])
            std = jnp.sqrt(variances[i])

            case_0 = lambda _: y + half_spread
            case_1 = lambda _: y - half_spread

            def case_2(_):
                lower = (y - alpha[i] - eta[i]) / std
                upper = (y + alpha[i] - eta[i]) / std
                eps = jr.truncated_normal(key, lower, upper)
                return eta[i] + std * eps

            return jax.lax.switch(event, [case_0, case_1, case_2], operand=None)

        auxiliaries = vmap(_sample_auxiliary)(
            auxiliary_keys,
            obs_values,
            event_types,
            bond_idxs,
            zs,
            etas,
        )

        eta_observed = etas[jnp.arange(etas.shape[0]), bond_idxs]
        residuals = auxiliaries - eta_observed

        ns = jnp.bincount(bond_idxs, length=D)
        residual_sums = jnp.bincount(
            bond_idxs,
            weights=residuals**2,
            length=D,
        )

        posterior_conc = concentration + 0.5 * ns
        posterior_scale = scale + 0.5 * residual_sums
        posterior = InverseGammaNatParam(alpha_plus_one=posterior_conc + 1, beta=posterior_scale)
        variances = posterior.dist_param.sample(variance_key)

        return {"R": jnp.diag(variances)}

    return ConditionalBlock(
        names=("R",),
        initialiser=_initialiser,
        kernel=_kernel,
    )


def _construct_Q0_block(D, concentration: float = 1.0, scale: float = 1.0):

    concentration = jnp.full((D,), concentration)
    scale = jnp.full((D,), scale)
    prior = InverseGammaNatParam(alpha_plus_one=concentration + 1, beta=scale)

    def _likelihood(context: GibbsContext):
        zs, _ = context.trajectory
        z0 = zs[0]
        return InverseGammaNatParam.from_gaussian(value=z0, mean=jnp.zeros((D,)))

    def _unpack(sample: Array):
        return {"Q0": jnp.diag(sample)}

    return ConjugateBlock(
        name="Q0",
        prior=prior,
        likelihood=_likelihood,
        unpack=_unpack,
    )


def _construct_Q_block(D, concentration: float = 1.0, scale: float = 1.0):

    concentration = jnp.full((D,), concentration)
    scale = jnp.full((D,), scale)
    prior = InverseGammaNatParam(alpha_plus_one=concentration + 1, beta=scale)

    def _likelihood(context: GibbsContext):
        zs, _ = context.trajectory
        A = jnp.diag(context.params["A"])                         # (D,)
        dts = context.dts[:, None]                                # (K-1, 1)

        means = jnp.exp(-dts * A[None, :]) * zs[:-1]              # (K-1, D)
        residuals = zs[1:] - means                                # (K-1, D)

        x = dts * A[None, :]
        safe_x = jnp.where(jnp.abs(x) < 1e-8, 1.0, x)
        ratios = -jnp.expm1(-2 * x) / (2 * safe_x)
        constants = dts * jnp.where(jnp.abs(x) < 1e-8, 1.0, ratios)

        likelihood_concentration = 0.5 * residuals.shape[0]
        likelihood_scale = 0.5 * jnp.sum(residuals**2 / constants, axis=0)

        return InverseGammaNatParam(
            alpha_plus_one=jnp.full((D,), likelihood_concentration),
            beta=likelihood_scale,
        )

    def _unpack(sample: Array):
        return {"Q": jnp.diag(sample)}

    return ConjugateBlock(
        name="Q",
        prior=prior,
        likelihood=_likelihood,
        unpack=_unpack,
    )