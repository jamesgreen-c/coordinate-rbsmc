import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.bayesian.kernels import KernelType
from experiments.bayesian.prior import _construct_cov_cholesky


parser = argparse.ArgumentParser()

parser.add_argument("--T", type=int, default=100)
parser.add_argument("--D", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)

parser.add_argument("--burnin", type=int, default=500)
parser.add_argument("--samples", type=int, default=500)

parser.add_argument("--phi", type=float, default=0.8)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)

parser.add_argument("--kernel", type=int, default=KernelType.CSMC.value)
parser.add_argument("--style", default="bootstrap")

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

args = parser.parse_args()

########################
#       Load data      #
########################
kernel_type = KernelType(args.kernel)

experiment_name = "kernel={},D={},T={},phi={},log-var={},N={},samples={},burnin={},conditional={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name, 
    args.D, 
    args.T, 
    args.phi, 
    args.log_var, 
    args.N,
    args.samples, 
    args.burnin, 
    args.conditional, 
    args.seed
)

dirpath = f"results/{experiment_name}"
datapath = f"{dirpath}/data.npz"

if not os.path.exists(datapath):
    raise FileNotFoundError(f"Could not find saved data at {datapath}")

plotpath = f"{dirpath}/plots"
os.makedirs(plotpath, exist_ok=True)

results = np.load(datapath, allow_pickle=True)
print(f"Loaded results from: {dirpath}")

# load posterior estimates and truth
param_hist = results["param_hist"].item()
beta_hist = np.asarray(param_hist["prior"]["trainable"]["beta"])
delta_hist = np.asarray(param_hist["prior"]["trainable"]["delta"])
true_beta = results["true_beta"]
true_delta = results["true_delta"]

sample_hist = results["sample_hist"]
true_xs = results["xs"]


########################
#       Plot data      #
########################
posterior_slice = slice(args.burnin + 1, args.burnin + args.samples + 1)

# evaluate prior parameter inference
beta_mean = beta_hist[posterior_slice].mean(axis=0)
delta_mean = delta_hist[posterior_slice].mean(axis=0)

true_chol_H = np.asarray(_construct_cov_cholesky(true_beta, true_delta))
chol_H_samples = np.stack([
    np.asarray(_construct_cov_cholesky(beta, delta))
    for beta, delta in zip(beta_hist[posterior_slice], delta_hist[posterior_slice])
])
chol_H_mean = chol_H_samples.mean(axis=0)

print("\nPosterior mean beta:\n", beta_mean)
print("True beta:\n", true_beta)
print("Beta absolute error:", np.abs(beta_mean - true_beta).sum())

print("\nPosterior mean delta:\n", delta_mean)
print("True delta:\n", true_delta)
print("Delta absolute error:", np.abs(delta_mean - true_delta).sum())

print("\nPosterior mean chol_H:\n", chol_H_mean)
print("True chol_H:\n", true_chol_H)
print("chol_H absolute error:", np.abs(chol_H_mean - true_chol_H).sum())


# plot loss over iterations
loss_history = results["loss_history"]

plt.figure()
plt.plot(loss_history)
plt.axvline(args.burnin, linestyle="--", color="black")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig(f"{plotpath}/loss_history.png", dpi=200, bbox_inches="tight")
plt.close()


# beta traces
fig, axes = plt.subplots(args.D, args.D, figsize=(3 * args.D, 2.5 * args.D), squeeze=False)

for j in range(args.D):
    for k in range(args.D):
        ax = axes[j, k]

        if k >= j:
            ax.axis("off")
            continue

        ax.plot(beta_hist[:, j, k])
        ax.axhline(true_beta[j, k], linestyle=":", color="red")
        ax.axvline(args.burnin, linestyle="--", color="black")
        ax.set_title(f"beta[{j},{k}]")

plt.tight_layout()
plt.savefig(f"{plotpath}/beta_traces.png", dpi=200, bbox_inches="tight")
plt.close()


# delta traces
fig, axes = plt.subplots(args.D, 1, figsize=(8, 2.5 * args.D), squeeze=False)

for d in range(args.D):
    ax = axes[d, 0]
    ax.plot(delta_hist[:, d])
    ax.axhline(true_delta[d], linestyle=":", color="red")
    ax.axvline(args.burnin, linestyle="--", color="black")
    ax.set_title(f"delta[{d}]")

plt.tight_layout()
plt.savefig(f"{plotpath}/delta_traces.png", dpi=200, bbox_inches="tight")
plt.close()


# state posterior estimates
sample_zs, sample_etas = sample_hist
true_zs, true_etas = true_xs

posterior_zs = sample_zs[posterior_slice]
posterior_etas = sample_etas[posterior_slice]

i = min(args.i, posterior_etas.shape[1] - 1)
d = min(args.component, args.D - 1)

mean_z = posterior_zs[:, i, :, d].mean(axis=0)
mean_eta = posterior_etas[:, i, :, d].mean(axis=0)

for name, samples, mean, truth in [
    ("z", posterior_zs, mean_z, true_zs[:, d]),
    ("eta", posterior_etas, mean_eta, true_etas[:, d]),
]:
    plt.figure()
    plt.plot(truth, label=f"true {name}", linestyle="--", color="blue")
    plt.plot(mean, label="posterior mean", color="black")

    for s in range(min(args.n_paths, args.samples)):
        plt.plot(samples[s, i, :, d], alpha=0.15, color="grey")

    plt.xlabel("t")
    plt.ylabel(f"{name}[{d}]")
    plt.legend()
    plt.savefig(f"{plotpath}/{name}_inference_i={i}_d={d}.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"{name}[{d}] posterior mean RMSE:", np.sqrt(np.mean((mean - truth) ** 2)))