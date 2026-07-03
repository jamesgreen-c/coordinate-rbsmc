import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from experiments.horseshoe.kernels import KernelType


# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--n-series", dest="n_series", type=int, default=2048)

parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
parser.add_argument("--num-iter", dest="num_iter", type=int, default=10_000)
parser.add_argument("--n-samples", dest="n_samples", type=int, default=64)

parser.add_argument("--T", dest="T", type=int, default=100)
parser.add_argument("--D", dest="D", type=int, default=1)

parser.add_argument("--phi", type=float, default=0.8)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)

parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC.value)
parser.add_argument("--style", dest="style", default="bootstrap")

parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--N", dest="N", type=int, default=31)

parser.add_argument("--i", dest="i", type=int, default=0)
parser.add_argument("--component", dest="component", type=int, default=0)
parser.add_argument("--n-paths", dest="n_paths", type=int, default=10)

args = parser.parse_args()


kernel_type = KernelType(args.kernel)

experiment_name = "kernel={},D={},T={},n-series={},phi={},log-var={},N={},n-samples={},batch-size={},num-iter={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name,
    args.D,
    args.T,
    args.n_series,
    args.phi,
    args.log_var,
    args.N,
    args.n_samples,
    args.batch_size,
    args.num_iter,
    args.seed,
)

dirpath = f"results/{experiment_name}"
datapath = f"{dirpath}/data.npz"

if not os.path.exists(datapath):
    raise FileNotFoundError(f"Could not find saved data at {datapath}")

plotpath = f"{dirpath}/plots"
if not os.path.exists(plotpath):
    os.mkdir(plotpath)


# load data
data = np.load(datapath, allow_pickle=True)
print(f"Loaded results from: {dirpath}")

# learned parameter error
final_parameters = data["final_params"].item()
best_parameters = data["best_params"].item()
final_A = final_parameters["prior"]["trainable"]["grad"]["A"]
best_A = best_parameters["prior"]["trainable"]["grad"]["A"]
true_A = data["true_A"]

final_error = np.abs(final_A - true_A).sum()
best_error = np.abs(best_A - true_A).sum()

print(f"\nFinal A estimation error: {final_error:.4f}")
print(f"Best A estimation error: {best_error:.4f}")
print(f"\nFinal estimated A:\n{final_A}")
print(f"\nBest estimated A:\n{best_A}")
print(f"\nTrue A:\n{true_A}")

# loss history
loss_history = data["loss_history"]

plt.figure()
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig(f"{plotpath}/loss_history.png", dpi=200, bbox_inches="tight")
plt.close()

# log loss history
plt.figure()
plt.plot(np.log(loss_history))
plt.xlabel("Iteration")
plt.ylabel("log loss")
plt.savefig(f"{plotpath}/log_loss_history.png", dpi=200, bbox_inches="tight")
plt.close()

# posterior inference plot
test_xs = data["test_xs"]
test_zs, test_etas = test_xs

test_data = data["test_data"]
test_ys = test_data[0]

test_samples = data["test_samples"]
test_sample_zs, test_sample_etas = test_samples

test_means = test_sample_etas.mean(axis=1)

B, S, T, D = test_sample_etas.shape
i = min(args.i, B - 1)
d = min(args.component, D - 1)

# choose sign based on RMSE 
truth = test_etas[i, :, d]
rmse_pos = np.sqrt(np.mean((test_means[i, :, d] - truth) ** 2))
rmse_neg = np.sqrt(np.mean((-test_means[i, :, d] - truth) ** 2))

sign = -1.0 if rmse_neg < rmse_pos else 1.0
test_means = sign * test_means
test_samples = sign * test_samples

# plot
plt.figure()
plt.plot(test_etas[i, :, d], label="true x", linestyle="--", color="blue")
# plt.scatter(np.arange(test_ys.shape[1]), test_ys[i, :, d], label="y", alpha=0.4, color="red", marker="x")
plt.plot(test_means[i, :, d], label="posterior mean", color="black")

for s in range(min(args.n_paths, S)):
    plt.plot(test_sample_etas[i, s, :, d], alpha=0.15, color="grey")

plt.xlabel("t")
plt.ylabel(f"x[{d}]")
plt.legend()
plt.savefig(f"{plotpath}/inference_i={i}_d={d}.png", dpi=200, bbox_inches="tight")
plt.close()


# mean replacement rate over training
if "replacement_rates" in data:
    replacement_rates = np.asarray(data["replacement_rates"])

    # Handles both current storage: (num_iter, T), and older scalar storage: (num_iter,)
    if replacement_rates.ndim == 2:
        mean_replacement_rates = np.nanmean(replacement_rates, axis=1)
    elif replacement_rates.ndim == 1:
        mean_replacement_rates = replacement_rates
    else:
        raise ValueError(f"Unexpected replacement_rates shape: {replacement_rates.shape}")

    plt.figure()
    plt.plot(mean_replacement_rates)
    plt.xlabel("Iteration")
    plt.ylabel("Mean replacement rate")
    plt.ylim(0.0, 1.0)
    plt.savefig(f"{plotpath}/mean_replacement_rate.png", dpi=200, bbox_inches="tight")
    plt.close()

    # plt.figure()
    # plt.plot(np.nanmean(replacement_rates[-500:], axis=0))
    # plt.xlabel("t")
    # plt.ylabel("Replacement rate")
    # plt.ylim(0.0, 1.0)
    # plt.savefig(f"{plotpath}/replacement_rate_profile_last500.png", dpi=200, bbox_inches="tight")
    # plt.close()
