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

    def estimate_R_and_alpha(self) -> tuple[Array, Array]:
        """Estimate R and D2D half-widths in the dataset's current units."""
        obs_values, bond_idxs, event_types = map(np.asarray, self.data)
        variances = np.full(self.D, np.nan)
        alphas = np.full(self.D, np.nan)

        for d in range(self.D):
            bond_mask = bond_idxs == d
            values = obs_values[bond_mask]
            types = event_types[bond_mask]

            previous, current = types[:-1], types[1:]
            switches = ((previous == 0) & (current == 1)) | (
                (previous == 1) & (current == 0)
            )
            n_switches = np.count_nonzero(switches)

            if n_switches:
                spread = np.abs(np.diff(values)[switches]).mean()
                alphas[d] = 0.25 * spread       # 0.05 * spread
                variances[d] = 0.05 * spread    # (0.05 * spread) ** 2
                flag = " [only one switch]" if n_switches == 1 else ""
                print(
                    f"Bond {d}: switches={n_switches}, spread={spread:.6g}, "
                    f"R[{d}, {d}]={variances[d]:.6g}, "
                    f"alpha[{d}]={alphas[d]:.6g}{flag}"
                )
            else:
                buy_sell_values = values[(types == 0) | (types == 1)]
                if buy_sell_values.size < 2:
                    print(f"Bond {d}: zero switches and fewer than two buy/sell observations")
                    continue

                variances[d] = buy_sell_values.var(ddof=1)
                alphas[d] = 0.05 * np.sqrt(variances[d])
                print(
                    f"Bond {d}: zero switches, "
                    f"sample variance={variances[d]:.6g}, "
                    f"alpha[{d}]={alphas[d]:.6g}"
                )

        invalid = np.flatnonzero(
            ~np.isfinite(variances) | ~np.isfinite(alphas)
        )
        if invalid.size:
            raise ValueError(f"Could not estimate R and alpha for bonds {invalid.tolist()}")

        R = jnp.diag(jnp.asarray(variances, dtype=obs_values.dtype))
        ALPHA = jnp.asarray(alphas, dtype=obs_values.dtype)
        return R, ALPHA
    
    # def estimate_R_and_alpha(self) -> Array:
    #     """Estimate diagonal emission covariance in the dataset's current units."""
    #     obs_values, bond_idxs, event_types = map(np.asarray, self.data)
    #     variances = np.full(self.D, np.nan)

    #     for d in range(self.D):
    #         bond_mask = bond_idxs == d
    #         values = obs_values[bond_mask]
    #         types = event_types[bond_mask]

    #         previous, current = types[:-1], types[1:]
    #         switches = ((previous == 0) & (current == 1)) | (
    #             (previous == 1) & (current == 0)
    #         )
    #         n_switches = np.count_nonzero(switches)

    #         if n_switches:
    #             spread = np.abs(np.diff(values)[switches]).mean()
    #             variances[d] = (0.05 * spread) ** 2
    #             flag = " [only one switch]" if n_switches == 1 else ""
    #             print(
    #                 f"Bond {d}: switches={n_switches}, spread={spread:.6g}, "
    #                 f"R[{d}, {d}]={variances[d]:.6g}{flag}"
    #             )
    #         else:
    #             buy_sell_values = values[(types == 0) | (types == 1)]
    #             if buy_sell_values.size < 2:
    #                 print(f"Bond {d}: zero switches and fewer than two buy/sell observations")
    #                 continue

    #             variances[d] = buy_sell_values.var(ddof=1)
    #             print(f"Bond {d}: zero switches, sample variance={variances[d]:.6g}")

    #     invalid = np.flatnonzero(~np.isfinite(variances))
    #     if invalid.size:
    #         raise ValueError(f"Could not estimate emission variance for bonds {invalid.tolist()}")

    #     return jnp.diag(jnp.asarray(variances, dtype=obs_values.dtype))