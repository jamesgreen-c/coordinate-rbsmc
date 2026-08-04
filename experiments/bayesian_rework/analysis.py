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

parser.add_argument("--phi", type=float, default=0.8)
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

args = parser.parse_args()

########################
#       Load data      #
########################

experiment_name = "kernel={},D={},T={},phi={},log-var={},N={},samples={},burnin={},conditional={},seed={}"
experiment_name = experiment_name.format(
    args.kernel, 
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
true_params = results["true_params"].item()
param_hist = results["params"].item()
sample_hist = results["trajectories"]
true_xs = results["xs"]


########################
#       Plot data      #
########################
posterior_slice = slice(args.burnin + 1, args.burnin + args.samples + 1)

# evaluate prior parameter inference
true_H = true_params["H"]
H_hist = param_hist["H"]
H_mean = H_hist[posterior_slice].mean(axis=0)
print("\nPosterior mean H:\n", H_mean)
print("True H:\n", true_H)
print("H absolute error:", np.abs(H_mean - true_H).sum())


# # plot loss over iterations
# loss_history = results["loss_history"]

# plt.figure()
# plt.plot(loss_history)
# plt.axvline(args.burnin, linestyle="--", color="black")
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.savefig(f"{plotpath}/loss_history.png", dpi=200, bbox_inches="tight")
# plt.close()


# H traces
fig, axes = plt.subplots(args.D, args.D, figsize=(3 * args.D, 2.5 * args.D), squeeze=False)

for j in range(args.D):
    for k in range(args.D):
        ax = axes[j, k]

        if k >= j:
            ax.axis("off")
            continue

        ax.plot(H_hist[:, j, k])
        ax.axhline(true_H[j, k], linestyle=":", color="red")
        ax.axvline(args.burnin, linestyle="--", color="black")
        ax.set_title(f"H[{j},{k}]")

plt.tight_layout()
plt.savefig(f"{plotpath}/H_traces.png", dpi=200, bbox_inches="tight")
plt.close()

# state posterior estimates
sample_zs, sample_etas = sample_hist
true_zs, true_etas = true_xs

posterior_zs = sample_zs[posterior_slice]
posterior_etas = sample_etas[posterior_slice]

d = min(args.component, args.D - 1)

mean_z = posterior_zs[:, :, d].mean(axis=0)
mean_eta = posterior_etas[:, :, d].mean(axis=0)

for name, samples, mean, truth in [
    ("z", posterior_zs, mean_z, true_zs[:, d]),
    ("eta", posterior_etas, mean_eta, true_etas[:, d]),
]:
    plt.figure()
    plt.plot(truth, label=f"true {name}", linestyle="--", color="blue")
    plt.plot(mean, label="posterior mean", color="black")

    for s in range(min(args.n_paths, samples.shape[0])):
        plt.plot(samples[s, :, d], alpha=0.15, color="grey")

    plt.xlabel("t")
    plt.ylabel(f"{name}[{d}]")
    plt.legend()
    plt.savefig(f"{plotpath}/{name}_inference_d={d}.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"{name}[{d}] posterior mean RMSE:", np.sqrt(np.mean((mean - truth) ** 2)))