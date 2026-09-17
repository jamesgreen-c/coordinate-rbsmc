import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--T", type=int, default=100)
parser.add_argument("--D", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--burnin", type=int, default=500)
parser.add_argument("--samples", type=int, default=500)
parser.add_argument("--thin", type=int, default=1)
parser.add_argument("--saved-paths", type=int, default=None)
parser.add_argument("--phi", type=float, default=0.1)
parser.add_argument("--kernel", type=str, default="CSMC")
parser.add_argument("--seed", type=int, default=1234)

parser.add_argument("--full-inference", action="store_true")
parser.add_argument("--no-full-inference", dest="full_inference", action="store_false")
parser.set_defaults(full_inference=False)

parser.add_argument("--conditional", action="store_true")
parser.add_argument("--unconditional", dest="conditional", action="store_false")
parser.set_defaults(conditional=True)

parser.add_argument("--backward", action="store_true")
parser.add_argument("--no-backward", dest="backward", action="store_false")
parser.set_defaults(backward=True)
parser.add_argument(
    "--backward-mode",
    choices=("ancestral", "ffbsi", "reduced"),
    default=None,
    help="Must match the path-selection mode used by experiment.py.",
)

parser.add_argument("--N", type=int, default=31)
parser.add_argument("--i", type=int, default=0)
parser.add_argument("--component", type=int, default=0)
parser.add_argument("--n-paths", dest="n_paths", type=int, default=10)

args = parser.parse_args()
backward_mode = args.backward_mode or ("ffbsi" if args.backward else "ancestral")


def get_pcs(precision):
    """Return partial correlations from a precision matrix."""
    diagonal = np.diag(precision)
    if np.any(diagonal <= 0):
        raise ValueError("Precision matrix must have positive diagonal entries.")

    pcs = -precision / np.sqrt(np.outer(diagonal, diagonal))
    np.fill_diagonal(pcs, 1.0)
    return pcs


def plot_traces(name, history, iterations, plotpath, burnin, thin, truth=None, lower_triangle=False):
    """Plot retained samples against their actual Gibbs iteration numbers."""
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
        fig, axes = plt.subplots(parameter_shape[0], 1, figsize=(7, 2.5 * parameter_shape[0]), squeeze=False)
        indices = [(i, 0, (i,)) for i in range(parameter_shape[0])]
    elif len(parameter_shape) == 2:
        fig, axes = plt.subplots(*parameter_shape, figsize=(3 * parameter_shape[1], 2.5 * parameter_shape[0]), squeeze=False)
        indices = [(i, j, (i, j)) for i in range(parameter_shape[0]) for j in range(parameter_shape[1])]
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
        if truth is not None:
            ax.axhline(np.asarray(truth)[index], linestyle=":", color="red")
        ax.axvline(burnin, linestyle="--", color="black")
        suffix = "".join(f"[{value}]" for value in index)
        ax.set_title(f"{name}{suffix}")
        ax.set_xlabel("Gibbs iteration")

    retention_text = (
        "Unthinned history"
        if thin == 1
        else f"Retained every {thin} Gibbs iterations"
    )
    fig.suptitle(f"{name} traces — {retention_text}")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{plotpath}/{name}_traces.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_summary(name, history, truth, posterior_selector, plotpath):
    """Plot true and posterior mean summaries for a vector or matrix parameter."""
    posterior_mean = history[posterior_selector].mean(axis=0)
    truth = np.asarray(truth)

    print(f"\nPosterior mean {name}:\n", posterior_mean)
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
        for ax, value, title in [(axes[0], true_value, f"True {name}"), (axes[1], posterior_value, f"Posterior mean {name}")]:
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
        for ax, value, title in [(axes[0], true_value, f"True {name}"), (axes[1], posterior_value, f"Posterior mean {name}")]:
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
        ("pcs", get_pcs(np.linalg.inv(truth)), np.stack([get_pcs(value) for value in precision_hist[posterior_selector]]).mean(axis=0), "partial correlation"),
    ]

    for label, true_value, posterior_value, colourbar_label in diagnostics:
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.05))
        axes = np.asarray([fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])])
        colourbar_axis = fig.add_subplot(grid[0, 2])
        scale = max(np.max(np.abs(true_value)), np.max(np.abs(posterior_value)))
        for ax, value, title in [(axes[0], true_value, f"True {label} {name}"), (axes[1], posterior_value, f"Posterior mean {label} {name}")]:
            image = ax.imshow(value, cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest")
            ax.set_title(title)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_xticks(np.arange(value.shape[1]))
            ax.set_yticks(np.arange(value.shape[0]))
        fig.colorbar(image, cax=colourbar_axis).set_label(colourbar_label)
        fig.savefig(f"{plotpath}/{label}_{name}_heatmaps.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

experiment_name = "kernel={},D={},T={},steps={},phi={},N={},samples={},burnin={},full-inference={},conditional={},seed={},backward-mode={}"
experiment_name = experiment_name.format(
    args.kernel,
    args.D,
    args.T,
    args.steps,
    args.phi,
    args.N,
    args.samples,
    args.burnin,
    args.full_inference,
    args.conditional,
    args.seed,
    backward_mode,
)

dirpath = f"results/{experiment_name}"
datapath = f"{dirpath}/data.npz"
if not os.path.exists(datapath):
    unthinned_name = "kernel={},D={},T={},steps={},phi={},N={},samples={},burnin={},full-inference={},conditional={},seed={},backward-mode={}"
    unthinned_name = unthinned_name.format(
        args.kernel,
        args.D,
        args.T,
        args.steps,
        args.phi,
        args.N,
        args.samples,
        args.burnin,
        args.full_inference,
        args.conditional,
        args.seed,
        backward_mode,
    )
    legacy_name = "kernel={},D={},T={},steps={},phi={},N={},samples={},burnin={},full-inference={},conditional={},seed={}"
    legacy_name = legacy_name.format(
        args.kernel,
        args.D,
        args.T,
        args.steps,
        args.phi,
        args.N,
        args.samples,
        args.burnin,
        args.full_inference,
        args.conditional,
        args.seed,
    )
    candidates = [
        (f"results/{unthinned_name}", f"results/{unthinned_name}/data.npz"),
        (f"results/{legacy_name}", f"results/{legacy_name}/data.npz"),
    ]
    for candidate_dir, candidate_path in candidates:
        if os.path.exists(candidate_path):
            dirpath = candidate_dir
            datapath = candidate_path
            break
    else:
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
burnin = int(saved_config.get("burnin", args.burnin))
samples = int(saved_config.get("samples", args.samples))
if thin < 1:
    raise ValueError(f"Saved thin value must be positive; got {thin}.")

num_retained_iterations = (burnin + samples - 1) // thin + 1
retained_iterations = np.arange(num_retained_iterations, dtype=int) * thin
parameter_iterations = np.concatenate((np.array([-1]), retained_iterations))
posterior_parameter_mask = parameter_iterations >= burnin

print(f"History thinning: every {thin} Gibbs iteration(s)")
print(f"Saved path limit: {saved_paths}")
print(
    f"Retained Gibbs iterations: {retained_iterations[0]} to "
    f"{retained_iterations[-1]} ({retained_iterations.size} values)"
)

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
estimated_params = results["estimated_params"].item()
param_hist = results["params"].item()
if reference_history is not None and "trajectory" in reference_history:
    sample_hist = reference_history["trajectory"]
elif "trajectories" in results.files:
    # Backward compatibility with result files written before Reference became
    # the single persisted SMC state.
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
posterior_reference_mask = reference_iterations >= burnin
if not np.any(posterior_parameter_mask):
    raise ValueError("No retained parameter samples fall after burn-in.")
if not np.any(posterior_reference_mask):
    raise ValueError("No retained reference paths fall after burn-in.")


##########################
#       Plot params      #
##########################
# Add each inferred parameter once here, including its inverse standardisation.
parameter_histories = {
    "m0": means[None, :] + scales[None, :] * param_hist["m0"],
    "H0": scales[None, :, None] * param_hist["H0"] * scales[None, None, :],
    "H": scales[None, :, None] * param_hist["H"] * scales[None, None, :],
}

for name, history in parameter_histories.items():
    truth = true_params[name]
    is_covariance = history.ndim == 3 and history.shape[-1] == history.shape[-2]
    plot_traces(
        name,
        history,
        parameter_iterations,
        plotpath,
        burnin,
        thin,
        truth,
        lower_triangle=is_covariance,
    )
    plot_posterior_summary(name, history, truth, posterior_parameter_mask, plotpath)

    if name in ("H", "H0"):
        plot_covariance_diagnostics(name, history, truth, posterior_parameter_mask, plotpath)

if "tau" in param_hist:
    plot_traces(
        "tau",
        param_hist["tau"],
        parameter_iterations,
        plotpath,
        burnin,
        thin,
    )

if "llambda" in param_hist:
    plot_traces(
        "llambda",
        param_hist["llambda"],
        parameter_iterations,
        plotpath,
        burnin,
        thin,
        lower_triangle=True,
    )


#################################
#       plot trajectories       #
#################################
sample_zs, sample_etas_standard = sample_hist
true_zs, true_etas = true_xs
sample_etas = means[None, None, :] + scales[None, None, :] * sample_etas_standard
posterior_zs = sample_zs[posterior_reference_mask]
posterior_etas = sample_etas[posterior_reference_mask]
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
        plt.plot(mean, label="posterior mean", color="black")
        for s in range(min(args.n_paths, samples.shape[0])):
            plt.plot(samples[s, :, d], alpha=0.15, color="grey")
        plt.xlabel("t")
        plt.ylabel(f"{name}[{d}]")
        plt.legend()
        plt.savefig(f"{state_plotpath}/{name}_inference_d={d}.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"{name}[{d}] posterior mean RMSE:", np.sqrt(np.mean((mean - truth) ** 2)))


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
posterior_replacement_mask = retained_iterations >= burnin
posterior_replacement_rates = replacement_rates[posterior_replacement_mask]
mean_replacement_rate = replacement_rates.mean(axis=1)

plt.figure()
plt.plot(
    retained_iterations,
    mean_replacement_rate,
    marker="o" if thin > 1 else None,
    markersize=2.5,
    linewidth=0.8,
)
plt.axvline(burnin, linestyle="--", color="black")
plt.xlabel("Gibbs iteration")
plt.ylabel("Mean replacement rate")
plt.ylim(0.0, 1.0)
plt.title(f"Retained every {thin} Gibbs iteration(s)")
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
plt.axvline(burnin, linestyle="--", color="white")
plt.xlabel("Gibbs iteration")
plt.ylabel("State time")
plt.title(f"Retained every {thin} Gibbs iteration(s)")
plt.colorbar(image).set_label("Replacement rate")
plt.tight_layout()
plt.savefig(f"{plotpath}/replacement_rate_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(posterior_replacement_rates.mean(axis=0), color="black")
plt.xlabel("State time")
plt.ylabel("Posterior mean replacement rate")
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig(f"{plotpath}/replacement_rate_by_time.png", dpi=200, bbox_inches="tight")
plt.close()
print("Posterior mean replacement rate:", posterior_replacement_rates.mean())


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
    plt.axvline(burnin, linestyle="--", color="white")
    plt.xlabel("Gibbs iteration")
    plt.ylabel("State time")
    path_text = "all retained paths" if saved_paths is None else f"latest {reference_iterations.size} paths"
    plt.title(f"{path_text}; retained every {thin} iteration(s)")
    plt.colorbar().set_label("Selected particle index")
    plt.tight_layout()
    plt.savefig(f"{plotpath}/reference_particle_indices.png", dpi=200, bbox_inches="tight")
    plt.close()
