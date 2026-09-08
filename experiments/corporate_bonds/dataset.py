import jax.numpy as jnp
import numpy as np

from jax import Array
from scipy.optimize import minimize
from scipy.linalg import solve_continuous_lyapunov

from rbsmc.utils.dataset import Dataset


################################
#       dataset override       # 
################################

class CorporateBondDataset(Dataset):
    data: tuple[Array, ...]
    states: tuple[Array, Array]
    params: dict[str, Array]
    CBBT: Array


    def __init__(
            self, 
            D: int, 
            dts: Array, 
            cbbt: Array,
            standardised: bool = False, 
            means: Array | None = None, 
            stds: Array | None = None, 
            **kwargs
        ):
        self.dts = dts
        self.CBBT = cbbt
        self.D = D
        self.standardised = standardised
        self.means = means
        self.stds = stds
        super().__init__(**kwargs)

    @property
    def standardised_data(self):
        """
        Return a new dataset standardised using an observation-based estimate
        of each bond's mid-YtB diffusion standard deviation.
        """
        if self.standardised:
            return self

        obs_values, bond_idxs, event_types = self.data

        obs_values_np = np.asarray(obs_values)
        bond_idxs_np = np.asarray(bond_idxs)
        event_types_np = np.asarray(event_types)
        dts_np = np.asarray(self.dts)

        times = np.concatenate((np.zeros(1), np.cumsum(dts_np)))

        means = np.zeros(self.D)
        scales = np.zeros(self.D)

        for d in range(self.D):
            indices = np.flatnonzero(bond_idxs_np == d)

            if len(indices) < 3:
                raise ValueError(f"Not enough observations to standardise bond {d}")

            observations = obs_values_np[indices]
            means[d] = observations.mean()

            left, right = np.triu_indices(len(indices), k=1)
            left_indices = indices[left]
            right_indices = indices[right]

            # use equal event types to reduce buy/sell spread offsets
            mask = event_types_np[left_indices] == event_types_np[right_indices]

            elapsed = times[right_indices] - times[left_indices]
            squared_differences = (obs_values_np[right_indices] - obs_values_np[left_indices])**2

            elapsed = elapsed[mask]
            squared_differences = squared_differences[mask]

            # fall back to all same-bond pairs if event matching is too sparse
            if len(elapsed) < 10:
                elapsed = times[right_indices] - times[left_indices]
                squared_differences = (obs_values_np[right_indices] - obs_values_np[left_indices])**2

            valid = np.isfinite(elapsed) & np.isfinite(squared_differences) & (elapsed > 0)
            elapsed = elapsed[valid]
            squared_differences = squared_differences[valid]

            if len(elapsed) < 2:
                raise ValueError(f"Not enough valid observation pairs to standardise bond {d}")

            # fit E[(Y_t - Y_s)^2] = intercept + H_dd * (t - s)
            centred_elapsed = elapsed - elapsed.mean()
            denominator = np.sum(centred_elapsed**2)
            variance = np.sum(centred_elapsed * (squared_differences - squared_differences.mean())) / denominator

            # a negative slope can occur with sparse or noisy observations;
            # use long-lag differences as a data-derived fallback
            if not np.isfinite(variance) or variance <= 0:
                long_lag = elapsed >= np.median(elapsed)
                variance = np.median(squared_differences[long_lag] / elapsed[long_lag]) / 0.4549364231

            if not np.isfinite(variance) or variance <= 0:
                raise ValueError(f"Could not estimate a positive diffusion variance for bond {d}")

            scales[d] = np.sqrt(variance)

        means = jnp.asarray(means, dtype=obs_values.dtype)
        stds = jnp.asarray(scales, dtype=obs_values.dtype)
        inv_stds = 1 / stds

        std_obs_values = (obs_values - means[bond_idxs]) / stds[bond_idxs]
        std_cbbt = (self.CBBT - means[bond_idxs]) / stds[bond_idxs]

        eta, z = self.states
        std_states = ((eta - means) / stds, z)

        std_params = {
            **self.params,
            "m0": inv_stds * (self.params["m0"] - means),
            "H0": inv_stds[:, None] * self.params["H0"] * inv_stds[None, :],
            "H": inv_stds[:, None] * self.params["H"] * inv_stds[None, :],
            "R": inv_stds[:, None] * self.params["R"] * inv_stds[None, :],
            "psi": self.params["psi"] / stds,
            "alpha": self.params["alpha"] / stds,
        }

        return CorporateBondDataset(
            D=self.D,
            dts=self.dts,
            data=(std_obs_values, bond_idxs, event_types),
            states=std_states,
            params=std_params,
            cbbt=std_cbbt,
            standardised=True,
            means=means,
            stds=stds,
        )



#########################################
#       estimate params from data       # 
#########################################
def estimate_params_from_data(dataset: CorporateBondDataset, alpha_scale: float = 1.0, H0_scale: float = 1.0):
    """
    Initialise model parameters from standardised transaction and CBBT data.

    Q is restricted to diagonal because each bond is fitted independently.
    H0 is calibrated rather than statistically estimated.
    """
    # dataset = dataset.standardised_data if not dataset.standardised else dataset
    PSI, A, Q = _estimate_ou_params(dataset)
    # M0, H, H0 = _estimate_mid_params(dataset, PSI, H0_scale)

    ALPHA = alpha_scale * PSI
    Q0 = np.diag(np.diag(Q) / (2.0 * np.diag(A)))
    # Q0 = solve_continuous_lyapunov(A, Q)

    return {
        # "m0": jnp.asarray(M0),
        # "H0": jnp.asarray(H0),
        # "H": jnp.asarray(H),
        "Q0": jnp.asarray(Q0),
        "Q": jnp.asarray(Q),
        "A": jnp.asarray(A),
        "psi": jnp.asarray(PSI),
        "alpha": jnp.asarray(ALPHA),
    }


def _estimate_ou_params(dataset: CorporateBondDataset):
    """
    Fit independent OU processes to standardised transaction-CBBT
    half-spread proxies.
    """

    def _objective(theta, times, x):
        log_A, log_Q = theta

        A = np.exp(log_A)
        Q = np.exp(log_Q)

        Q0 = Q / (2.0 * A)
        loss = 0.5 * (np.log(2.0 * np.pi * Q0) + x[0]**2 / Q0)

        elapsed = np.diff(times)
        F = np.exp(-A * elapsed)
        Q_k = Q * (1.0 - np.exp(-2.0 * A * elapsed)) / (2.0 * A)
        residuals = x[1:] - F * x[:-1]

        return loss + 0.5 * np.sum(
            np.log(2.0 * np.pi * Q_k) + residuals**2 / Q_k
        )

    std_obs_values, bond_idxs, event_types = dataset.data

    trades = np.asarray(std_obs_values)
    bond_idxs = np.asarray(bond_idxs)
    std_cbbt = np.asarray(dataset.CBBT)
    dts = np.asarray(dataset.dts)
    times = np.concatenate((np.zeros(1), np.cumsum(dts)))

    D = dataset.D

    A = np.zeros((D, D))
    Q = np.zeros((D, D))
    PSI = np.zeros(D)

    for d in range(D):
        indices = np.flatnonzero(bond_idxs == d)

        bond_trades = trades[indices]
        bond_cbbt = std_cbbt[indices]
        bond_times = times[indices]

        psi_proxy = np.abs(bond_trades - bond_cbbt)
        valid = np.isfinite(psi_proxy) & np.isfinite(bond_times) & (psi_proxy > 0)

        psi_proxy = psi_proxy[valid]
        bond_times = bond_times[valid]

        if len(psi_proxy) < 3:
            raise ValueError(f"Not enough valid spread proxies for bond {d}")

        log_psi = np.log(np.maximum(psi_proxy, 1e-8))
        log_PSI_d = np.median(log_psi)
        x = log_psi - log_PSI_d

        total_time = bond_times[-1] - bond_times[0]
        min_A = 1.0 / total_time

        theta_init = np.array([
            np.log(max(1.0, min_A)),
            np.log(0.1),
        ])

        result = minimize(
            _objective,
            theta_init,
            args=(bond_times, x),
            method="L-BFGS-B",
            bounds=[
                (np.log(min_A), np.log(1e3)),
                (np.log(1e-8), np.log(1e3)),
            ],
        )

        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"OU estimation failed for bond {d}: {result.message}")
        
        log_A_d, log_Q_d = result.x

        PSI[d] = np.exp(log_PSI_d)
        A[d, d] = np.exp(log_A_d)
        Q[d, d] = np.exp(log_Q_d)

    # print(f"PSI: {PSI}")
    # print(f"A: {A}")
    # print(f"Q: {Q}")

    return PSI, A, Q


# def _estimate_mid_params(dataset: CorporateBondDataset, PSI: np.ndarray, H0_scale: float):
#     """
#     Initialise M0 and estimate diagonal H from standardised CBBT observations.

#     H0 is calibrated to represent H0_scale typical half-spreads of initial
#     uncertainty in each coordinate.
#     """
#     std_obs_values, bond_idxs, event_types = dataset.data

#     bond_idxs = np.asarray(bond_idxs)
#     cbbt = np.asarray(dataset.CBBT)
#     dts = np.asarray(dataset.dts)
#     times = np.concatenate((np.zeros(1), np.cumsum(dts)))

#     M0 = np.zeros(dataset.D)
#     H = np.zeros((dataset.D, dataset.D))

#     for d in range(dataset.D):
#         indices = np.flatnonzero((bond_idxs == d) & np.isfinite(cbbt))

#         if len(indices) < 2:
#             raise ValueError(f"Not enough finite CBBT observations for bond {d}")

#         M0[d] = cbbt[indices[0]]

#         elapsed = np.diff(times[indices])
#         increments = np.diff(cbbt[indices])
#         valid = np.isfinite(elapsed) & np.isfinite(increments) & (elapsed > 0)

#         elapsed = elapsed[valid]
#         increments = increments[valid]

#         if len(elapsed) == 0:
#             raise ValueError("No valid CBBT increments for bond {}".format(d))

#         H[d, d] = np.mean(increments**2 / elapsed)

#     if np.any(~np.isfinite(np.diag(H))) or np.any(np.diag(H) <= 0):
#         raise ValueError("Could not estimate a positive diagonal H")

#     H0 = np.diag((H0_scale * PSI)**2)

#     return M0, H, H0
