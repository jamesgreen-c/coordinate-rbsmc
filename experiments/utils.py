import numpy as np
import jax.numpy as jnp
from jax import Array

################################
#       helper functions       # 
################################

def ou_diag_transition(A, Q, dt):
    """
    Exact OU transition for dz = -diag(A_diag) z dt + chol_Q dB_t.

    Parameters
    ----------
    A:       (D, D) Diagonal transition matrix
    chol_Q:  Cholesky factor of the covariance
    
    Returns
    -------
    F:   (D, D) transition matrix
    Cov: (D, D) transition covariance
    """

    A_diag = jnp.diag(A)
    a_sum = A_diag[:, None] + A_diag[None, :]
    factor = jnp.where(
        jnp.abs(a_sum) > 1e-10,
        (1.0 - jnp.exp(-a_sum * dt)) / a_sum,
        dt,
    )

    F = jnp.diag(jnp.exp(-A_diag * dt))
    Cov = factor * Q
    chol_Cov = jnp.linalg.cholesky(Cov)
    return F, chol_Cov

def _diag_or_vector_at(chol_R: Array, i: Array):
    """
    Returns the scalar observation standard deviation for bond i.

    Parameters
    ----------
    chol_R: (dim,) or (dim, dim)
    """
    if chol_R.ndim == 1:
        return chol_R[i]
    elif chol_R.ndim == 2:
        return chol_R[i, i]
    else:
        raise ValueError("chol_R must have shape (dim,) or (dim, dim).")

def _logdiffexp(a: Array, b: Array):
    """ Computes log(exp(a) - exp(b)), assuming a >= b. """
    # return a + jnp.log1p(-jnp.exp(b - a))
    return a + jnp.log1p(-jnp.exp(jnp.minimum(b - a, -1e-7)))


def print_z_diagnostics(dataset):
    obs_values, bond_idxs, event_types = map(np.asarray, dataset.data)
    true_zs, true_etas = map(np.asarray, dataset.states)

    psi = np.asarray(dataset.params["psi"])
    R = np.asarray(dataset.params["R"])
    H = np.asarray(dataset.params["H"])

    times = np.concatenate((np.zeros(1), np.cumsum(np.asarray(dataset.dts))))
    true_spreads = psi[None, :] * np.exp(true_zs)

    print("\n========================")
    print("z inference diagnostics")
    print("========================")

    for d in range(dataset.D):
        indices = np.flatnonzero(bond_idxs == d)
        types_d = event_types[indices]
        spreads_d = true_spreads[indices, d]
        zs_d = true_zs[indices, d]

        noise_std = np.sqrt(R[d, d])
        spread_mean = spreads_d.mean()
        spread_std = spreads_d.std()

        # local sensitivity dy/dz = psi * exp(z)
        z_sensitivity = spreads_d
        fisher_information = z_sensitivity**2 / R[d, d]

        counts = np.bincount(types_d, minlength=5)

        print(f"\nBond {d}")
        print("  number of observations:", len(indices))
        print("  event counts [buy, sell, TA buy, TA sell, D2D]:", counts)
        print("  true z mean:", zs_d.mean())
        print("  true z std:", zs_d.std())
        print("  mean spread:", spread_mean)
        print("  spread std:", spread_std)
        print("  observation-noise std:", noise_std)
        print("  mean spread / noise std:", spread_mean / noise_std)
        print("  spread variation / noise std:", spread_std / noise_std)
        print("  mean direct-observation Fisher information:", fisher_information.mean())
        print("  approximate single-observation z std:", 1 / np.sqrt(fisher_information.mean()))

        if len(indices) > 1:
            gaps = np.diff(times[indices])
            eta_increment_std = np.sqrt(H[d, d] * np.median(gaps))

            print("  median time between observations:", np.median(gaps))
            print("  eta movement over median gap:", eta_increment_std)
            print("  eta movement / spread variation:", eta_increment_std / spread_std)

            eta_predictive_variance = H[d, d] * np.median(gaps)
            effective_variance = eta_predictive_variance + R[d, d]
            effective_information = spreads_d**2 / effective_variance

            print("  eta predictive variance:", eta_predictive_variance)
            print("  effective observation variance:", effective_variance)
            print("  effective z information:", effective_information.mean())
            print("  effective single-observation z std:", 1 / np.sqrt(effective_information.mean()))

        # check whether the direct-trade emission residuals match R
        direct_buy = indices[types_d == 0]
        direct_sell = indices[types_d == 1]

        buy_residuals = obs_values[direct_buy] - true_etas[direct_buy, d] + true_spreads[direct_buy, d]
        sell_residuals = obs_values[direct_sell] - true_etas[direct_sell, d] - true_spreads[direct_sell, d]
        direct_residuals = np.concatenate((buy_residuals, sell_residuals))

        if len(direct_residuals) > 1:
            print("  empirical direct-trade noise std:", direct_residuals.std())
            print("  expected direct-trade noise std:", noise_std)

        # distance between opposite-side direct observations
        direct = indices[np.isin(types_d, [0, 1])]

        if len(direct) > 1:
            opposite_gaps = []

            for left in range(len(direct)):
                for right in range(left + 1, len(direct)):
                    if event_types[direct[left]] != event_types[direct[right]]:
                        opposite_gaps.append(times[direct[right]] - times[direct[left]])

            if opposite_gaps:
                print("  median opposite-side trade gap:", np.median(opposite_gaps))
            else:
                print("  no opposite-side direct-trade pairs")

    print("\n")