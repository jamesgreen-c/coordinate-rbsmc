import jax.numpy as jnp
import numpy as np

from jax import Array
from rbsmc.utils.dataset import Dataset


################################
#       dataset override       # 
################################

class CorporateBondDataset(Dataset):
    data: tuple[Array, ...]
    states: Array
    params: dict[str, Array]


    def __init__(self, D: int, dts: Array, **kwargs):
        self.dts = dts
        self.D = D
        self.means = None
        self.stds = None
        super().__init__(**kwargs)

    # override standardisation
    # @property
    # def standardised_data(self):
    #     """
    #     Standardise each observation using the mean and standard deviation
    #     of the observations belonging to the corresponding bond.
    #     """
    #     obs_values, bond_idxs, event_types = self.data

    #     counts = jnp.bincount(bond_idxs, length=self.D)

    #     if bool(jnp.any(counts == 0)):
    #         missing = jnp.where(counts == 0)[0]
    #         raise ValueError(f"Cannot standardise bonds with no observations: {missing}")

    #     sums = jnp.zeros(self.D, dtype=obs_values.dtype).at[bond_idxs].add(obs_values)
    #     self.means = sums / counts

    #     centred_obs = obs_values - self.means[bond_idxs]
    #     squared_sums = jnp.zeros(self.D, dtype=obs_values.dtype).at[bond_idxs].add(centred_obs**2)
    #     self.stds = jnp.sqrt(squared_sums / counts)

    #     if bool(jnp.any(self.stds == 0)):
    #         constant = jnp.where(self.stds == 0)[0]
    #         raise ValueError(f"Cannot standardise bonds with zero observation variance: {constant}")

    #     std_obs_values = centred_obs / self.stds[bond_idxs]

    #     return std_obs_values, bond_idxs, event_types

    @property
    def standardised_data(self):
        """
        Standardise each bond using an observation-based estimate of its
        mid-YtB diffusion standard deviation.
        """
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

        self.means = jnp.asarray(means, dtype=obs_values.dtype)
        self.stds = jnp.asarray(scales, dtype=obs_values.dtype)

        std_obs_values = (obs_values - self.means[bond_idxs]) / self.stds[bond_idxs]

        return std_obs_values, bond_idxs, event_types

    @property
    def standardised_params(self):
        """ 
        Standardise all params.
        Log half-spread dynamics are not standardised as they are unitless:
            - The rescaling occurs through psi = psi / stds.
        """
        # TODO move means and std calculation to init? 
        assert self.means is not None and self.stds is not None, "Standardise data first"

        inv_stds = 1 / self.stds

        # extract
        M0 = self.params["m0"]
        H0 = self.params["H0"]
        H = self.params["H"]
        R = self.params["R"]
        PSI = self.params["psi"]
        ALPHA = self.params["alpha"]

        # standardise
        M0 = inv_stds * (M0 - self.means)
        H0 = inv_stds[:, None] * H0 * inv_stds[None, :]
        H = inv_stds[:, None] * H * inv_stds[None, :]
        R = inv_stds[:, None] * R * inv_stds[None, :]
        PSI = PSI / self.stds
        ALPHA = ALPHA / self.stds
        
        standardised_params = {
            **self.params,
            "m0": M0,
            "H0": H0,
            "H": H,
            "R": R,
            "psi": PSI,
            "alpha": ALPHA,
        }
        return standardised_params