import jax.numpy as jnp
import jax.random as jr

from jax.scipy.linalg import solve_triangular
from jax.scipy.special import multigammaln


class InverseWhishart:

    @classmethod
    def sample(cls, key, df, scale):
        """
        Sample Sigma ~ IW(df, scale), where

            p(Sigma) ∝ |Sigma|^{-(df + D + 1) / 2} * exp(-0.5 * tr(scale @ Sigma^{-1})).

        Parameters
        ----------
        key:    JAX PRNG key.
        df:     Degrees of freedom, requiring df > D - 1.
        scale:  Positive-definite inverse-Wishart scale matrix, shape (D, D).

        Returns
        -------
        Sigma:  Positive-definite sample, shape (D, D).
        """
        D = scale.shape[-1]
        key_diag, key_lower = jr.split(key)

        # Bartlett diagonal: B[i, i]^2 ~ chi-square(df - i)
        dfs = df - jnp.arange(D)
        diag = jnp.sqrt(2.0 * jr.gamma(key_diag, 0.5 * dfs))

        # independent standard normal in lower diagonal.
        lower = jnp.tril(jr.normal(key_lower, (D, D)), k=-1)
        B = lower + jnp.diag(diag)

        # solve for covariance
        L = jnp.linalg.cholesky(scale)
        X = solve_triangular(B, L.T, lower=True)
        Sigma = X.T @ X
        return 0.5 * (Sigma + Sigma.T)
    
    @classmethod
    def logpdf(cls, Sigma, df, scale, *args, **kwargs):
        """
        Evaluate log p(Sigma) for Sigma ~ IW(df, scale).

        Requires
            df > D - 1,
            Sigma positive definite,
            scale positive definite.
        """
        D = Sigma.shape[-1]

        chol_Sigma = jnp.linalg.cholesky(Sigma)
        chol_scale = jnp.linalg.cholesky(scale)

        logdet_Sigma = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol_Sigma, axis1=-2, axis2=-1)), axis=-1)
        logdet_scale = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol_scale, axis1=-2, axis2=-1)), axis=-1)

        # Sigma^{-1} scale, calculated through triangular solves.
        tmp = solve_triangular(chol_Sigma, scale, lower=True)
        Sigma_inv_scale = solve_triangular(chol_Sigma.mT, tmp, lower=False)
        trace_term = jnp.trace(Sigma_inv_scale, axis1=-2, axis2=-1)

        log_normalizer = (0.5*df*logdet_scale - 0.5*df*D*jnp.log(2.0) - multigammaln(0.5 * df, D))

        return (
            log_normalizer
            - 0.5 * (df + D + 1.0) * logdet_Sigma
            - 0.5 * trace_term
        )
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from jax import vmap
    
    # x-axis
    covariance_values = jnp.linspace(0.01, 3.0, 1000)
    Sigmas = covariance_values[:, None, None]

    def evaluate_pdf(df, scale):
        scale_matrix = jnp.array([[scale]])
        logpdfs = vmap(lambda Sigma: InverseWhishart.logpdf(Sigma=Sigma, df=df, scale=scale_matrix))(Sigmas)
        return jnp.exp(logpdfs)
    
    # axes
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))

    # vary df while holding scale fixed.
    dfs = [3.0, 5.0, 10.0]
    fixed_scale = 1.0

    for df in dfs:
        pdfs = evaluate_pdf(df=df, scale=fixed_scale)
        ax[0].plot(covariance_values, pdfs, label=rf"$\nu={df:g}$")

    ax[0].set_xlabel(r"Covariance $\Sigma$")
    ax[0].set_ylabel(r"Density $p(\Sigma)$")
    ax[0].set_title(rf"Inverse-Wishart density with fixed scale $\Lambda={fixed_scale:g}$")
    ax[0].legend()
    ax[0].grid(alpha=0.25)

    # vary scale while holding df fixed.
    scales = [0.5, 1.0, 2.0]
    fixed_df = 5.0

    for scale in scales:
        pdfs = evaluate_pdf(df=fixed_df, scale=scale)
        ax[1].plot(covariance_values, pdfs, label=rf"$\Lambda={scale:g}$")

    ax[1].set_xlabel(r"Covariance $\Sigma$")
    ax[1].set_ylabel(r"Density $p(\Sigma)$")
    ax[1].set_title(rf"Inverse-Wishart density with fixed df $\nu={fixed_df:g}$")
    ax[1].legend()
    ax[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig("inverse-wishart-density.png", dpi=200)