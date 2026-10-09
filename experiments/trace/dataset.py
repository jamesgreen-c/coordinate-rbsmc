import jax.numpy as jnp
import numpy as np

from jax import Array


################################
#       dataset override       # 
################################

class TraceDataset:

    def __init__(
            self, 
            D: int, 
            dts: Array,
            data: tuple[Array],
            standardised: bool = False, 
            means: Array | None = None, 
            stds: Array | None = None, 
            **kwargs
        ):
        self.data = data
        self.dts = dts
        self.D = D
        self.standardised = standardised
        self.means = means
        self.stds = stds

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

        std_obs_values = (obs_values - means[bond_idxs]) / stds[bond_idxs]

        return TraceDataset(
            D=self.D,
            dts=self.dts,
            data=(std_obs_values, bond_idxs, event_types),
            standardised=True,
            means=means,
            stds=stds,
        )

