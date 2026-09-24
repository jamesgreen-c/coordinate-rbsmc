import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--T", type=int, default=500)
parser.add_argument("--D", type=int, default=1)
parser.add_argument("--steps", type=int, default=499)
parser.add_argument("--num-iter", type=int, default=1000)
parser.add_argument("--thin", type=int, default=1)
parser.add_argument("--saved-paths", type=int, default=None)
parser.add_argument("--phi", type=float, default=0.1)
parser.add_argument("--kernel", type=str, default="CSMC")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--backward-mode", choices=("ancestral", "ffbsi", "reduced"), default=None)

parser.add_argument("--full-inference", action="store_true")
parser.add_argument("--no-full-inference", dest="full_inference", action="store_false")
parser.set_defaults(full_inference=False)

parser.add_argument("--conditional", action="store_true")
parser.add_argument("--unconditional", dest="conditional", action="store_false")
parser.set_defaults(conditional=True)

parser.add_argument("--backward", action="store_true")
parser.add_argument("--no-backward", dest="backward", action="store_false")
parser.set_defaults(backward=True)

parser.add_argument("--N", type=int, default=31)
parser.add_argument("--i", type=int, default=0)
parser.add_argument("--component", type=int, default=0)
parser.add_argument("--n-paths", dest="n_paths", type=int, default=10)
parser.add_argument(
    "--summary-start",
    type=int,
    default=None,
    help="First iteration included in summaries; by default, detect an energy plateau.",
)

args = parser.parse_args()

backward_mode = args.backward_mode or ("ffbsi" if args.backward else "ancestral")


def energy_plateau_start(energies):
    """Find the earliest sustained energy plateau using nonoverlapping block means."""
    num_iter = len(energies)
    block_size = max(10, num_iter // 25)
    if num_iter < 6 * block_size:
        return max(0, num_iter - max(1, num_iter // 4)), False
    blocks = [energies[i:i + block_size] for i in range(0, num_iter, block_size)]
    block_means = np.array([block.mean() for block in blocks])
    final_mean = block_means[-4:].mean()
    late_noise = np.median(
        [block.std(ddof=1) / np.sqrt(len(block))
         for block in blocks[-4:] if len(block) > 1]
    )
    tolerance = max(3 * late_noise, 0.005 * abs(final_mean))

    # All remaining block means must be close to the final level.
    for block in range(max(1, len(blocks) // 10), len(blocks) - 3):
        if np.all(np.abs(block_means[block:] - final_mean) <= tolerance):
            return block * block_size, True
    return num_iter - max(1, num_iter // 4), False


def get_pcs(precision):
    """Return partial correlations from a precision matrix."""
    diagonal = np.diag(precision)
    if np.any(diagonal <= 0):
        raise ValueError("Precision matrix must have positive diagonal entries.")
    pcs = -precision / np.sqrt(np.outer(diagonal, diagonal))
    np.fill_diagonal(pcs, 1.0)
    return pcs


def plot_traces(
    name, history, iterations, plotpath, thin, truth=None,
    lower_triangle=False, summary_start=None,
):
    """Plot retained samples against their actual iteration numbers."""
    history = np.asarray(history)
    iterations = np.asarray(iterations)
    if history.shape[0] != iterations.size:
        raise ValueError(
            f"{name} history has {history.shape[0]} entries but received "
            f"{iterations.size} iteration indices."
        )
    parameter_shape = history.shape[1:]
    if len(parameter_shape) == 0:
        fig, axes = plt.subplots(1, 1)
        axes = np.asarray([[axes]])
        indices = [(0, 0, ())]

    elif len(parameter_shape) == 1:
        fig, axes = plt.subplots(
            parameter_shape[0], 1, figsize=(7, 2.5 * parameter_shape[0]), squeeze=False
        )
        indices = [(i, 0, (i,)) for i in range(parameter_shape[0])]

    elif len(parameter_shape) == 2:
        fig, axes = plt.subplots(
            *parameter_shape,
            figsize=(3 * parameter_shape[1], 2.5 * parameter_shape[0]),
            squeeze=False,
        )
        indices = [
            (i, j, (i, j))
            for i in range(parameter_shape[0])
            for j in range(parameter_shape[1])
        ]

    else:
        raise ValueError(f"{name} must be scalar, vector, or matrix-valued; got shape {parameter_shape}.")

    for i, j, index in indices:
        ax = axes[i, j]

        if lower_triangle and len(index) == 2 and index[1] > index[0]:
            ax.axis("off")
            continue

        ax.plot(
            iterations,
            history[(slice(None),) + index],
            marker="o" if thin > 1 else None,
            markersize=2.5,
            linewidth=0.8,
        )

        if summary_start is not None:
            ax.axvline(summary_start, linestyle="--", color="black", linewidth=0.8)

        if truth is not None:
            ax.axhline(np.asarray(truth)[index], linestyle=":", color="red")

        suffix = "".join(f"[{value}]" for value in index)
        ax.set_title(f"{name}{suffix}")
        ax.set_xlabel("Iteration")

    retention_text = "Unthinned history" if thin == 1 else f"Retained every {thin} iterations"
    fig.suptitle(f"{name} traces — {retention_text}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{plotpath}/{name}_traces.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_summary(name, history, truth, posterior_selector, plotpath):
    """Plot true and late-iteration mean summaries for a vector or matrix parameter."""
    posterior_mean = history[posterior_selector].mean(axis=0)
    truth = np.asarray(truth)

    print(f"\nLate-iteration mean {name}:\n", posterior_mean)
    print(f"True {name}:\n", truth)
    print(f"{name} absolute error:", np.abs(posterior_mean - truth).sum())

    if truth.ndim == 1:
        true_value = np.atleast_2d(truth)
        posterior_value = np.atleast_2d(posterior_mean)

        fig = plt.figure(figsize=(12, 3.5), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.12))
        axes = np.asarray([fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])])
        colourbar_axis = fig.add_subplot(grid[1, :])
        scale = max(np.max(np.abs(true_value)), np.max(np.abs(posterior_value)))

        panels = [
            (axes[0], true_value, f"True {name}"),
            (axes[1], posterior_value, f"Late-iteration mean {name}"),
        ]
        for ax, value, title in panels:
            image = ax.imshow(value, cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("Component")
            ax.set_xticks(np.arange(value.shape[1]))
            ax.set_yticks([])

        fig.colorbar(image, cax=colourbar_axis, orientation="horizontal").set_label(f"{name} value")

    elif truth.ndim == 2:
        true_value = np.atleast_2d(truth)
        posterior_value = np.atleast_2d(posterior_mean)

        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.05))
        axes = np.asarray([fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])])
        colourbar_axis = fig.add_subplot(grid[0, 2])
        scale = max(np.max(np.abs(true_value)), np.max(np.abs(posterior_value)))

        panels = [
            (axes[0], true_value, f"True {name}"),
            (axes[1], posterior_value, f"Late-iteration mean {name}"),
        ]
        for ax, value, title in panels:
            image = ax.imshow(value, cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest")
            ax.set_title(title)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_xticks(np.arange(value.shape[1]))
            ax.set_yticks(np.arange(value.shape[0]))

        fig.colorbar(image, cax=colourbar_axis).set_label(f"{name} value")

    else:
        raise ValueError(f"{name} must be vector or matrix-valued; got shape {truth.shape}.")

    fig.savefig(f"{plotpath}/{name}_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_covariance_diagnostics(name, history, truth, posterior_selector, plotpath):
    """Plot precision and partial-correlation summaries for a covariance parameter."""
    precision_hist = np.linalg.inv(history)
    diagnostics = [
        ("precision", np.linalg.inv(truth), precision_hist[posterior_selector].mean(axis=0), "precision value"),
        (
            "pcs",
            get_pcs(np.linalg.inv(truth)),
            np.stack([get_pcs(value) for value in precision_hist[posterior_selector]]).mean(axis=0),
            "partial correlation",
        ),
    ]

    for label, true_value, posterior_value, colourbar_label in diagnostics:
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.05))
        axes = np.asarray([fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])])
        colourbar_axis = fig.add_subplot(grid[0, 2])
        scale = max(np.max(np.abs(true_value)), np.max(np.abs(posterior_value)))

        panels = [
            (axes[0], true_value, f"True {label} {name}"),
            (axes[1], posterior_value, f"Late-iteration mean {label} {name}"),
        ]
        for ax, value, title in panels:
            image = ax.imshow(value, cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest")
            ax.set_title(title)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_xticks(np.arange(value.shape[1]))
            ax.set_yticks(np.arange(value.shape[0]))

        fig.colorbar(image, cax=colourbar_axis).set_label(colourbar_label)
        fig.savefig(f"{plotpath}/{label}_{name}_heatmaps.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

experiment_name = (
    "kernel={},D={},T={},steps={},phi={},N={},num-iter={},"
    "full-inference={},conditional={},seed={},backward-mode={}"
)
experiment_name = experiment_name.format(
    args.kernel,
    args.D,
    args.T,
    args.steps,
    args.phi,
    args.N,
    args.num_iter,
    args.full_inference,
    args.conditional,
    args.seed,
    backward_mode,
)

dirpath = f"results/{experiment_name}"
datapath = f"{dirpath}/data.npz"

if not os.path.exists(datapath):
    raise FileNotFoundError(f"Could not find saved data at {datapath}")

plotpath = f"{dirpath}/plots"
os.makedirs(plotpath, exist_ok=True)

results = np.load(datapath, allow_pickle=True)
print(f"Loaded results from: {dirpath}")
saved_config = results["config"].item() if "config" in results.files else {}

if not isinstance(saved_config, dict):
    saved_config = vars(saved_config)

thin = int(saved_config.get("thin", args.thin))
saved_paths = saved_config.get("saved_paths", args.saved_paths)
num_iter = int(saved_config.get("num_iter", args.num_iter))

if thin < 1:
    raise ValueError(f"Saved thin value must be positive; got {thin}.")

if num_iter < 1:
    raise ValueError("No completed iterations are available to analyse.")

retained_iterations = np.arange(0, num_iter, thin, dtype=int)

# Index zero is initialization; history index k + 1 is the state after iteration k.
parameter_iterations = np.concatenate((np.array([-1]), retained_iterations))

print(f"History thinning: every {thin} iteration(s)")
print(f"Saved path limit: {saved_paths}")
print(f"Retained iterations: {retained_iterations[0]} to {retained_iterations[-1]} ({retained_iterations.size} values)")

reference_history = results["references"].item() if "references" in results.files else None
if reference_history is not None:
    if "ancestors" in reference_history:
        reference_particle_indices = np.asarray(reference_history["ancestors"])

    elif "particle_indices" in reference_history:
        reference_particle_indices = np.asarray(reference_history["particle_indices"])

    else:
        reference_particle_indices = None

elif "reference_particle_indices" in results.files:
    reference_particle_indices = results["reference_particle_indices"]

elif "ancestors" in results.files:
    reference_particle_indices = results["ancestors"]

else:
    reference_particle_indices = None

true_params = results["true_params"].item()
param_hist = results["params"].item()

if reference_history is not None and "trajectory" in reference_history:
    sample_hist = reference_history["trajectory"]

elif "trajectories" in results.files:

    # backward compatibility with results before Reference became the single persisted SMC state.
    sample_hist = results["trajectories"]

else:
    raise KeyError("The result file contains neither a reference trajectory nor trajectories.")

dataset = results["dataset"].item()
true_xs = dataset.states
means = results["standardisation_means"]
scales = results["standardisation_scales"]
first_param_history = np.asarray(next(iter(param_hist.values())))

if first_param_history.shape[0] != parameter_iterations.size:
    raise ValueError(
        "Parameter-history length does not agree with the saved thinning "
        f"configuration: found {first_param_history.shape[0]}, expected "
        f"{parameter_iterations.size}."
    )

num_reference_paths = np.asarray(sample_hist[0]).shape[0]
full_reference_iterations = parameter_iterations
expected_reference_paths = (
    full_reference_iterations.size
    if saved_paths is None
    else min(int(saved_paths), full_reference_iterations.size)
)

if num_reference_paths != expected_reference_paths:
    raise ValueError(
        "Reference-history length does not agree with saved_paths: found "
        f"{num_reference_paths}, expected {expected_reference_paths}."
    )

if num_reference_paths > full_reference_iterations.size:
    raise ValueError(
        f"Reference history contains {num_reference_paths} paths, exceeding "
        f"the {full_reference_iterations.size} paths implied by the config."
    )

reference_iterations = full_reference_iterations[-num_reference_paths:]

# Energies are recorded on every iteration, independently of history thinning.
energies = np.asarray(results["energies"])
if energies.shape != (num_iter,) or not np.all(np.isfinite(energies)):
    raise ValueError(f"Expected {num_iter} finite energies; found shape {energies.shape}.")

if args.summary_start is None:
    summary_start, plateau_found = energy_plateau_start(energies)
    method = "energy plateau" if plateau_found else "final quarter (no clear plateau)"
else:
    summary_start = args.summary_start
    method = "manual cutoff"

if not 0 <= summary_start < num_iter:
    raise ValueError(f"summary-start must lie in [0, {num_iter - 1}].")

summary_parameter_mask = parameter_iterations >= summary_start
summary_reference_mask = reference_iterations >= summary_start
if not np.any(summary_parameter_mask):
    raise ValueError("No saved parameter states lie after the summary cutoff.")

if not np.any(summary_reference_mask):
    raise ValueError(
        "No saved paths lie after the summary cutoff; increase saved_paths "
        "or choose an earlier --summary-start."
    )
print(f"Summary starts at iteration {summary_start} ({method}).")
print(f"Parameter states in summary: {summary_parameter_mask.sum()}")
print(f"Reference paths in summary: {summary_reference_mask.sum()}")

# Reconstruct inferred parameters in their original units.
chol_H_hist = np.asarray(param_hist["chol_H"])
H_standard_hist = chol_H_hist @ np.swapaxes(chol_H_hist, -1, -2)
tau_hist = np.exp(np.asarray(param_hist["log_tau"]))
raw_llambda = np.exp(np.asarray(param_hist["log_llambda"]))
llambda_hist = np.triu(raw_llambda, k=1)
llambda_hist = llambda_hist + np.swapaxes(llambda_hist, -1, -2) + np.eye(raw_llambda.shape[-1])

parameter_histories = {

    # "m0": means[None, :] + scales[None, :] * np.asarray(param_hist["m0"]),

    # "H0": scales[None, :, None] * np.asarray(param_hist["H0"]) * scales[None, None, :],
    "H": scales[None, :, None] * H_standard_hist * scales[None, None, :],
}

for name, history in parameter_histories.items():
    truth = true_params[name]
    is_covariance = history.ndim == 3 and history.shape[-1] == history.shape[-2]
    plot_traces(
        name,
        history,
        parameter_iterations,
        plotpath,
        thin,
        truth,
        lower_triangle=is_covariance,
        summary_start=summary_start,
    )
    plot_posterior_summary(name, history, truth, summary_parameter_mask, plotpath)

    if name in ("H", "H0"):
        plot_covariance_diagnostics(name, history, truth, summary_parameter_mask, plotpath)

plot_traces("tau", tau_hist, parameter_iterations, plotpath, thin, summary_start=summary_start)
plot_traces(
    "llambda", llambda_hist, parameter_iterations, plotpath, thin,
    lower_triangle=True, summary_start=summary_start,
)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.arange(num_iter), energies, linewidth=0.8)
ax.axvline(summary_start, linestyle="--", color="black", label=f"Summary from {summary_start}")
ax.legend()
ax.set(xlabel="Iteration", ylabel="Energy", title="Energy history")
fig.tight_layout()
fig.savefig(f"{plotpath}/energy_trace.png", dpi=200, bbox_inches="tight")
plt.close(fig)

sample_zs, sample_etas_standard = sample_hist
true_zs, true_etas = true_xs

sample_etas = means[None, None, :] + scales[None, None, :] * sample_etas_standard
posterior_zs = sample_zs[summary_reference_mask]
posterior_etas = sample_etas[summary_reference_mask]

z_plotpath = f"{plotpath}/zs"
eta_plotpath = f"{plotpath}/etas"
os.makedirs(z_plotpath, exist_ok=True)
os.makedirs(eta_plotpath, exist_ok=True)

for d in range(args.D):
    for name, samples, truth, state_plotpath in [
        ("z", posterior_zs, true_zs[:, d], z_plotpath),
        ("eta", posterior_etas, true_etas[:, d], eta_plotpath),
    ]:
        mean = samples[:, :, d].mean(axis=0)

        plt.figure(figsize=(25, 5))
        plt.plot(truth, label=f"true {name}", linestyle="--", color="blue")
        plt.plot(mean, label="late-iteration mean", color="black")

        for s in range(min(args.n_paths, samples.shape[0])):
            plt.plot(samples[s, :, d], alpha=0.15, color="grey")

        plt.xlabel("t")
        plt.ylabel(f"{name}[{d}]")
        plt.legend()
        plt.savefig(f"{state_plotpath}/{name}_inference_d={d}.png", dpi=200, bbox_inches="tight")
        plt.close()

        print(f"{name}[{d}] late-iteration mean RMSE:", np.sqrt(np.mean((mean - truth) ** 2)))

####################################

#       plot replacement rate      #

####################################
replacement_rates = results["replacement_rates"]
if replacement_rates.shape[0] != retained_iterations.size:
    raise ValueError(
        "Replacement-rate history does not agree with the saved thinning "
        f"configuration: found {replacement_rates.shape[0]}, expected "
        f"{retained_iterations.size}."
    )

mean_replacement_rate = replacement_rates.mean(axis=1)
plt.figure()
plt.plot(
    retained_iterations,
    mean_replacement_rate,
    marker="o" if thin > 1 else None,
    markersize=2.5,
    linewidth=0.8,
)
plt.xlabel("Iteration")
plt.ylabel("Mean replacement rate")
plt.ylim(0.0, 1.0)
plt.title(f"Retained every {thin} iteration(s)")
plt.tight_layout()
plt.savefig(f"{plotpath}/replacement_rate_trace.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
half_width = thin / 2
image = plt.imshow(
    replacement_rates.T,
    aspect="auto",
    origin="lower",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    interpolation="nearest",
    extent=(
        retained_iterations[0] - half_width,
        retained_iterations[-1] + half_width,
        -0.5,
        replacement_rates.shape[1] - 0.5,
    ),
)
plt.xlabel("Iteration")
plt.ylabel("State time")
plt.title(f"Retained every {thin} iteration(s)")
plt.colorbar(image).set_label("Replacement rate")
plt.tight_layout()
plt.savefig(f"{plotpath}/replacement_rate_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(replacement_rates.mean(axis=0), color="black")
plt.xlabel("State time")
plt.ylabel("Late-iteration mean replacement rate")
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig(f"{plotpath}/replacement_rate_by_time.png", dpi=200, bbox_inches="tight")
plt.close()
print("Late-iteration mean replacement rate:", replacement_rates.mean())

#########################################

#       selected particle indices       #

#########################################
if reference_particle_indices is not None:
    if reference_particle_indices.shape[0] != reference_iterations.size:
        raise ValueError(
            "Reference-particle history and reference-iteration coordinates "
            f"have different lengths: {reference_particle_indices.shape[0]} "
            f"and {reference_iterations.size}."
        )

    plt.figure(figsize=(10, 5))
    half_width = thin / 2
    plt.imshow(
        reference_particle_indices.T,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        interpolation="nearest",
        extent=(
            reference_iterations[0] - half_width,
            reference_iterations[-1] + half_width,
            -0.5,
            reference_particle_indices.shape[1] - 0.5,
        ),
    )
    plt.xlabel("Iteration")
    plt.ylabel("State time")
    path_text = "all retained paths" if saved_paths is None else f"latest {reference_iterations.size} paths"
    plt.title(f"{path_text}; retained every {thin} iteration(s)")
    plt.colorbar().set_label("Selected particle index")
    plt.tight_layout()
    plt.savefig(f"{plotpath}/reference_particle_indices.png", dpi=200, bbox_inches="tight")
    plt.close()
