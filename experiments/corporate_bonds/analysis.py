"""
PLOT THE CORPORATE BOND SMOOTHING RESULTS
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.corporate_bonds.kernels import KernelType


# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=10)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--M", dest="M", type=int, default=5)
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--independent", action="store_true")
parser.set_defaults(independent=False)

parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC)
parser.add_argument("--style", dest="style", type=str, default="bootstrap")

parser.add_argument("--i", type=int, default=0)
args = parser.parse_args()


###############################
#  SINGLE ANALYSIS FUNCTIONS  #
###############################

# --- args ---
kernel_type = KernelType(args.kernel)
Ts = np.cumsum(np.repeat(args.T / args.steps, args.steps))

EVENT_LABELS = {
    0: "Done (Buy)",
    1: "Done (Sell)",
    2: "Traded Away (Buy)",
    3: "Traded Away (Sell)",
    4: "D2D Trade",
}

EVENT_MARKERS = {
    0: "o",
    1: "o",
    2: "+",
    3: "+",
    4: "x",
}

EVENT_COLORS = {
    0: "red",
    1: "green",
    2: "red",
    3: "green",
    4: "black",
}

QUANTILES = (
    (1, 99, 0.10, "1%-99%"),
    (5, 95, 0.14, "5%-95%"),
    (10, 90, 0.18, "10%-90%"),
    (25, 75, 0.24, "25%-75%"),
)


# --- functions ---
def _legend(fig, axes):
    """
    Create a single de-duplicated legend for all axes.
    """
    handles, labels = [], []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)

    seen = set()
    handles_unique, labels_unique = [], []
    for handle, label in zip(handles, labels):
        if label not in seen and not label.startswith("_"):
            handles_unique.append(handle)
            labels_unique.append(label)
            seen.add(label)

    fig.legend(handles_unique, labels_unique, loc="upper center", ncol=5, frameon=False)


def _plot_observations_on_axis(ax, bond_indices, event_types, obs_values, dim):
    """
    Plot observations attached to a single bond index.
    """
    mask_dim = bond_indices == dim

    for event_type in EVENT_LABELS:
        mask = mask_dim & (event_types == event_type)
        if not np.any(mask):
            continue

        ax.scatter(
            Ts[mask],
            obs_values[mask],
            color=EVENT_COLORS[event_type],
            marker=EVENT_MARKERS[event_type],
            s=35,
            linewidths=1.25,
            label=EVENT_LABELS[event_type],
            zorder=5,
        )


def plot_observations(data, dirpath):
    """
    Plot one panel per bond showing only the relevant observed transaction or quote values.

    Parameters
    ----------
    data : dict-like
        Must contain:
        - bond_indices: shape (K, steps)
        - event_types:  shape (K, steps)
        - obs_values:   shape (K, steps)
    dirpath : str
        Directory where the figure will be saved.

    Returns
    -------
    None
    """

    bond_indices = np.asarray(data["bond_indices"][args.i]).astype(int)
    event_types = np.asarray(data["event_types"][args.i]).astype(int)
    obs_values = np.asarray(data["obs_values"][args.i])

    fig, axes = plt.subplots(args.D, 1, figsize=(14, 3.25 * args.D), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for dim, ax in enumerate(axes):
        _plot_observations_on_axis(ax, bond_indices, event_types, obs_values, dim)
        ax.set_title(f"Bond {dim + 1}")
        ax.set_ylabel("Observed YtB / quote")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Business time")
    _legend(fig, axes)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{dirpath}/observations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_estimations(data, dirpath):
    """
    Plot one panel per bond showing quantile-band smoothing estimates and the relevant observations.

    Notes
    -----
    The experiment stores smoothing paths, not filter particle clouds. Therefore the shaded regions are
    empirical marginal quantile envelopes over the sampled smoothing paths at each observation time.
    Individual smoothing samples are deliberately not plotted.

    Parameters
    ----------
    data : dict-like
        Must contain:
        - etas:         shape (K, M, steps, D)
        - true_etas:    shape (K, steps, D)
        - bond_indices: shape (K, steps)
        - event_types:  shape (K, steps)
        - obs_values:   shape (K, steps)
    dirpath : str
        Directory where the figure will be saved.

    Returns
    -------
    None
    """

    etas = np.asarray(data["etas"][args.i])
    true_etas = np.asarray(data["true_etas"][args.i])
    bond_indices = np.asarray(data["bond_indices"][args.i]).astype(int)
    event_types = np.asarray(data["event_types"][args.i]).astype(int)
    obs_values = np.asarray(data["obs_values"][args.i])

    _, _, D = etas.shape

    fig, axes = plt.subplots(D, 1, figsize=(14, 3.5 * D), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for dim, ax in enumerate(axes):
        paths_dim = etas[:, :, dim]

        for q_low, q_high, alpha, label in QUANTILES:
            lo, hi = np.percentile(paths_dim, [q_low, q_high], axis=0)
            ax.fill_between(Ts, lo, hi, color="blue", alpha=alpha, linewidth=0, label=label)

        median = np.median(paths_dim, axis=0)
        ax.plot(Ts, median, color="black", linestyle="--", linewidth=1.4, label="Median smoothing path")
        ax.plot(Ts, true_etas[:, dim], color="black", alpha=0.55, linewidth=1.2, label="True eta")

        _plot_observations_on_axis(ax, bond_indices, event_types, obs_values, dim)
        ax.set_title(f"Bond {dim + 1}")
        ax.set_ylabel("Mid-YtB eta")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Business time")
    _legend(fig, axes)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f"{dirpath}/estimations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


########################
#  load data function  #
########################

def load_data(kernel, style, D, steps):
    experiment_name = "kernel={},style={},D={},T={},N={},steps={},M={},independent={},seed={}"
    experiment_name = experiment_name.format(
        kernel.name,
        style,
        D,
        args.T,
        args.N,
        steps,
        args.M,
        args.independent,
        args.seed,
    )
    dirpath = f"results/{experiment_name}"
    if not os.path.exists(dirpath):
        print("No such experiment exists")
        print(experiment_name)
        exit()

    data = np.load(f"{dirpath}/data.npz")
    return data, dirpath


data, dirpath = load_data(kernel_type, args.style, args.D, args.steps)
plot_observations(data, dirpath)
plot_estimations(data, dirpath)



i = 0
zs = data["zs"][i]      # (M, steps, D)
etas = data["etas"][i]  # (M, steps, D)

print("finite zs:", np.isfinite(zs).all())
print("finite etas:", np.isfinite(etas).all())

print("max |z|:", np.max(np.abs(zs)))
print("max |eta|:", np.max(np.abs(etas)))

print("z quantiles:", np.quantile(zs, [0.0, 0.5, 0.9, 0.99, 0.999, 1.0]))
print("eta quantiles:", np.quantile(etas, [0.0, 0.5, 0.9, 0.99, 0.999, 1.0]))

m, t, d = np.unravel_index(np.argmax(np.abs(etas)), etas.shape)
print("worst eta index:", m, t, d)
print("worst eta:", etas[m, t, d])
print("corresponding z path:", zs[m, t])
print("event:")
print("bond_idx:", data["bond_indices"][i, t])
print("event_type:", data["event_types"][i, t])
print("obs_value:", data["obs_values"][i, t])