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
parser.add_argument("--phi", type=float, default=0.1)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)
parser.add_argument("--kernel", type=str, default="CSMC")
parser.add_argument("--seed", type=int, default=1234)

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

parser.add_argument("--infer-H", dest="infer_H", action='store_true')
parser.add_argument("--no-infer-H", dest="infer_H", action='store_false')
parser.set_defaults(infer_H=True)

parser.add_argument("--infer-m0", dest="infer_m0", action='store_true')
parser.add_argument("--no-infer-m0", dest="infer_m0", action='store_false')
parser.set_defaults(infer_m0=True)

parser.add_argument("--infer-H0", dest="infer_H0", action='store_true')
parser.add_argument("--no-infer-H0", dest="infer_H0", action='store_false')
parser.set_defaults(infer_H0=True)

args = parser.parse_args()


def get_pcs(precision):
    """Return partial correlations from a precision matrix."""
    diagonal = np.diag(precision)
    if np.any(diagonal <= 0):
        raise ValueError("Precision matrix must have positive diagonal entries.")

    pcs = -precision / np.sqrt(np.outer(diagonal, diagonal))
    np.fill_diagonal(pcs, 1.0)
    return pcs


def plot_traces(name, history, plotpath, burnin, truth=None, lower_triangle=False):
    """Plot traces for a scalar, vector, or matrix-valued parameter."""
    history = np.asarray(history)
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

        ax.plot(history[(slice(None),) + index])
        if truth is not None:
            ax.axhline(np.asarray(truth)[index], linestyle=":", color="red")
        ax.axvline(burnin, linestyle="--", color="black")
        suffix = "".join(f"[{value}]" for value in index)
        ax.set_title(f"{name}{suffix}")

    fig.tight_layout()
    fig.savefig(f"{plotpath}/{name}_traces.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_summary(name, history, truth, posterior_slice, plotpath):
    """Plot true and posterior mean summaries for a vector or matrix parameter."""
    posterior_mean = history[posterior_slice].mean(axis=0)
    truth = np.asarray(truth)

    print(f"\nPosterior mean {name}:\n", posterior_mean)
    print(f"True {name}:\n", truth)
    print(f"{name} absolute error:", np.abs(posterior_mean - truth).sum())

    if truth.ndim == 1:
        true_value = np.atleast_2d(truth)
        posterior_value = np.atleast_2d(posterior_mean)
        fig, axes = plt.subplots(1, 2, figsize=(12, 2.75))
        scale = max(np.max(np.abs(true_value)), np.max(np.abs(posterior_value)))
        for ax, value, title in [(axes[0], true_value, f"True {name}"), (axes[1], posterior_value, f"Posterior mean {name}")]:
            image = ax.imshow(value, cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("Component")
            ax.set_xticks(np.arange(value.shape[1]))
            ax.set_yticks([])
        fig.subplots_adjust(wspace=0.3, bottom=0.42)
        fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.10, pad=0.30).set_label(f"{name} value")
    elif truth.ndim == 2:
        true_value = np.atleast_2d(truth)
        posterior_value = np.atleast_2d(posterior_mean)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        scale = max(np.max(np.abs(true_value)), np.max(np.abs(posterior_value)))
        for ax, value, title in [(axes[0], true_value, f"True {name}"), (axes[1], posterior_value, f"Posterior mean {name}")]:
            image = ax.imshow(value, cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest")
            ax.set_title(title)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_xticks(np.arange(value.shape[1]))
            ax.set_yticks(np.arange(value.shape[0]))
        fig.subplots_adjust(wspace=0.3, right=0.88)
        fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04).set_label(f"{name} value")
    else:
        raise ValueError(f"{name} must be vector or matrix-valued; got shape {truth.shape}.")

    fig.savefig(f"{plotpath}/{name}_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_covariance_diagnostics(name, history, truth, posterior_slice, plotpath):
    """Plot precision and partial-correlation summaries for a covariance parameter."""
    precision_hist = np.linalg.inv(history)
    diagnostics = [
        ("precision", np.linalg.inv(truth), precision_hist[posterior_slice].mean(axis=0), "precision value"),
        ("pcs", get_pcs(np.linalg.inv(truth)), np.stack([get_pcs(value) for value in precision_hist[posterior_slice]]).mean(axis=0), "partial correlation"),
    ]

    for label, true_value, posterior_value, colourbar_label in diagnostics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        scale = max(np.max(np.abs(true_value)), np.max(np.abs(posterior_value)))
        for ax, value, title in [(axes[0], true_value, f"True {label} {name}"), (axes[1], posterior_value, f"Posterior mean {label} {name}")]:
            image = ax.imshow(value, cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest")
            ax.set_title(title)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_xticks(np.arange(value.shape[1]))
            ax.set_yticks(np.arange(value.shape[0]))
        fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04).set_label(colourbar_label)
        fig.subplots_adjust(wspace=0.3)
        fig.savefig(f"{plotpath}/{label}_{name}_heatmaps.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

# experiment_name = "kernel={},D={},T={},steps={},phi={},log-var={},N={},samples={},burnin={},conditional={},seed={}".format(
#     args.kernel, args.D, args.T, args.steps, args.phi, args.log_var, args.N,
#     args.samples, args.burnin, args.conditional, args.seed,
# )

# experiment_name = "kernel={},D={},T={},steps={},phi={},log-var={},N={},s={},b={},inf-H={},inf-m0={},cond={},seed={}"
# experiment_name = experiment_name.format(
#     args.kernel, 
#     args.D, 
#     args.T, 
#     args.steps, 
#     args.phi, 
#     args.log_var, 
#     args.N,
#     args.samples, 
#     args.burnin,
#     args.infer_H,
#     args.infer_m0,
#     args.conditional, 
#     args.seed
# )
experiment_name = "kernel={},D={},T={},steps={},phi={},log-var={},N={},s={},b={},inf-H={},inf-m0={},inf-H0={},cond={},seed={}"
experiment_name = experiment_name.format(
    args.kernel,
    args.D,
    args.T,
    args.steps,
    args.phi,
    args.log_var,
    args.N,
    args.samples,
    args.burnin,
    args.infer_H,
    args.infer_m0,
    args.infer_H0,
    args.conditional,
    args.seed,
)

dirpath = f"results/{experiment_name}"
datapath = f"{dirpath}/data.npz"
if not os.path.exists(datapath):
    raise FileNotFoundError(f"Could not find saved data at {datapath}")

plotpath = f"{dirpath}/plots"
os.makedirs(plotpath, exist_ok=True)
results = np.load(datapath, allow_pickle=True)
print(f"Loaded results from: {dirpath}")

true_params = results["true_params"].item()
param_hist = results["params"].item()
sample_hist = results["trajectories"]
dataset = results["dataset"].item()
true_xs = dataset.states
means = results["standardisation_means"]
scales = results["standardisation_scales"]
posterior_slice = slice(args.burnin + 1, args.burnin + args.samples + 1)


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
    plot_traces(name, history, plotpath, args.burnin, truth, lower_triangle=is_covariance)
    plot_posterior_summary(name, history, truth, posterior_slice, plotpath)

    if name in ("H", "H0"):
        plot_covariance_diagnostics(name, history, truth, posterior_slice, plotpath)

if "tau" in param_hist:
    plot_traces("tau", param_hist["tau"], plotpath, args.burnin)

if "llambda" in param_hist:
    plot_traces("llambda", param_hist["llambda"], plotpath, args.burnin, lower_triangle=True)


#################################
#       plot trajectories       #
#################################
sample_zs, sample_etas_standard = sample_hist
true_zs, true_etas = true_xs
sample_etas = means[None, None, :] + scales[None, None, :] * sample_etas_standard
posterior_zs = sample_zs[posterior_slice]
posterior_etas = sample_etas[posterior_slice]
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
posterior_replacement_rates = replacement_rates[args.burnin:args.burnin + args.samples]
mean_replacement_rate = replacement_rates.mean(axis=1)

plt.figure()
plt.plot(mean_replacement_rate)
plt.axvline(args.burnin, linestyle="--", color="black")
plt.xlabel("Gibbs iteration")
plt.ylabel("Mean replacement rate")
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig(f"{plotpath}/replacement_rate_trace.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
image = plt.imshow(replacement_rates.T, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
plt.axvline(args.burnin, linestyle="--", color="white")
plt.xlabel("Gibbs iteration")
plt.ylabel("State time")
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
