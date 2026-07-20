import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import solve_triangular

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
    def logpdf(cls):
        return None  # TODO