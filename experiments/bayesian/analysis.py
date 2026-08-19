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

args = parser.parse_args()

########################
#       Load data      #
########################

experiment_name = "kernel={},D={},T={},steps={},phi={},log-var={},N={},samples={},burnin={},conditional={},seed={}"
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
dataset = results["dataset"].item()
true_xs = dataset.states

# load the transformation used to standardise the inference problem
means = results["standardisation_means"]
scales = results["standardisation_scales"]


##########################
#       Plot params      #
##########################
posterior_slice = slice(args.burnin + 1, args.burnin + args.samples + 1)

# evaluate prior parameter inference
true_H = true_params["H"]
H_hist_standard = param_hist["H"]
H_hist = scales[None, :, None] * H_hist_standard * scales[None, None, :]
H_mean = H_hist[posterior_slice].mean(axis=0)
print("\nPosterior mean H:\n", H_mean)
print("True H:\n", true_H)
print("H absolute error:", np.abs(H_mean - true_H).sum())


# plot precision matrices
D = true_H.shape[0]
true_prec_H = np.linalg.inv(true_H)

if "beta" in param_hist:
    beta_hist_standard = param_hist["beta"]
    beta_hist = beta_hist_standard / scales[None, :, None] / scales[None, None, :]
    prec_H_mean = beta_hist[posterior_slice].mean(axis=0)

    inverse_error = np.max(np.abs(beta_hist[posterior_slice] - np.linalg.inv(H_hist[posterior_slice])))
    print("Maximum beta/H inverse error:", inverse_error)
else:
    prec_H_mean = np.linalg.inv(H_hist[posterior_slice]).mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
scale = max(np.max(np.abs(true_prec_H)), np.max(np.abs(prec_H_mean)))

for ax, H, title in [(axes[0], true_prec_H, "True precision H"), (axes[1], prec_H_mean, "Posterior mean precision H")]:
    image = ax.imshow(
        H,
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(args.D))
    ax.set_yticks(np.arange(args.D))

# shared colour bar for both matrices
colourbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
colourbar.set_label("prec H value")

# fig.subplots_adjust(left=0.07, right=0.88, bottom=0.10, top=0.90, wspace=0.25)
plt.savefig(f"{plotpath}/prec_H_heatmaps.png", dpi=200, bbox_inches="tight")
plt.close()


# plot partial correlations
def get_pcs(prec):
    diags = np.diag(prec)

    if np.any(diags <= 0):
        raise ValueError("Precision matrix must have positive diagonal entries.")

    pcs = -prec / np.sqrt(np.outer(diags, diags))
    np.fill_diagonal(pcs, 1.0)

    return pcs

true_pcs = get_pcs(np.asarray(true_prec_H))
posterior_beta = beta_hist[posterior_slice]
posterior_pcs_hist = np.stack([get_pcs(beta) for beta in posterior_beta])
posterior_pcs = posterior_pcs_hist.mean(axis=0)
# posterior_pcs = get_pcs(prec_H_mean)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
scale = max(np.max(np.abs(true_pcs)), np.max(np.abs(posterior_pcs)))

for ax, H, title in [(axes[0], true_pcs, "True precision pcs"), (axes[1], posterior_pcs, "Posterior mean pcs")]:
    image = ax.imshow(
        H,
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(args.D))
    ax.set_yticks(np.arange(args.D))

# shared colour bar for both matrices
colourbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
colourbar.set_label("pc value")

# fig.subplots_adjust(left=0.07, right=0.88, bottom=0.10, top=0.90, wspace=0.25)
plt.savefig(f"{plotpath}/pc_heatmaps.png", dpi=200, bbox_inches="tight")
plt.close()

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

        if k > j:
            ax.axis("off")
            continue

        ax.plot(H_hist[:, j, k])
        ax.axhline(true_H[j, k], linestyle=":", color="red")
        ax.axvline(args.burnin, linestyle="--", color="black")
        ax.set_title(f"H[{j},{k}]")

plt.tight_layout()
plt.savefig(f"{plotpath}/H_traces.png", dpi=200, bbox_inches="tight")
plt.close()


# true and posterior mean H heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# use one shared scale so colours are directly comparable
scale = max(np.max(np.abs(true_H)), np.max(np.abs(H_mean)))

for ax, H, title in [(axes[0], true_H, "True H"), (axes[1], H_mean, "Posterior mean H")]:
    image = ax.imshow(
        H,
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(args.D))
    ax.set_yticks(np.arange(args.D))

# shared colour bar for both matrices
colourbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
colourbar.set_label("H value")

fig.subplots_adjust(left=0.07, right=0.88, bottom=0.10, top=0.90, wspace=0.25)
plt.savefig(f"{plotpath}/H_heatmaps.png", dpi=200, bbox_inches="tight")
plt.close()


# tau trace
tau_hist = param_hist["tau"]
plt.figure()
plt.plot(tau_hist)
plt.title("Tau trace")
plt.axvline(args.burnin, linestyle="--", color="black")
plt.tight_layout()
plt.savefig(f"{plotpath}/tau_trace.png", dpi=200, bbox_inches="tight")
plt.close()


# llambda traces
llambda_hist = param_hist["llambda"]
fig, axes = plt.subplots(args.D, args.D, figsize=(3 * args.D, 2.5 * args.D), squeeze=False)

for j in range(args.D):
    for k in range(args.D):
        ax = axes[j, k]

        if k >= j:
            ax.axis("off")
            continue

        ax.plot(llambda_hist[:, j, k])
        ax.axvline(args.burnin, linestyle="--", color="black")
        ax.set_title(f"Lambda[{j},{k}]")

plt.tight_layout()
plt.savefig(f"{plotpath}/llambda_traces.png", dpi=200, bbox_inches="tight")
plt.close()


#################################
#       plot trajectories       #
#################################

# state posterior estimates
sample_zs, sample_etas_standard = sample_hist
true_zs, true_etas = true_xs

# transform the posterior mid-YtB trajectories back to their original scale
sample_etas = means[None, None, :] + scales[None, None, :] * sample_etas_standard

posterior_zs = sample_zs[posterior_slice]
posterior_etas = sample_etas[posterior_slice]
# d = min(args.component, args.D - 1)


z_plotpath = f"{plotpath}/zs"
eta_plotpath = f"{plotpath}/etas"

os.makedirs(z_plotpath, exist_ok=True)
os.makedirs(eta_plotpath, exist_ok=True)

print(posterior_zs.shape)
for d in range(args.D):
    mean_z = posterior_zs[:, :, d].mean(axis=0)
    mean_eta = posterior_etas[:, :, d].mean(axis=0)

    for name, samples, mean, truth, state_plotpath in [
        ("z", posterior_zs, mean_z, true_zs[:, d], z_plotpath),
        ("eta", posterior_etas, mean_eta, true_etas[:, d], eta_plotpath),
    ]:
        plt.figure(figsize=(25, 5))
        plt.plot(truth, label=f"true {name}", linestyle="--", color="blue")
        plt.plot(mean, label="posterior mean", color="black")

        for s in range(min(args.n_paths, samples.shape[0])):
            plt.plot(samples[s, :, d], alpha=0.15, color="grey")

        plt.xlabel("t")
        plt.ylabel(f"{name}[{d}]")
        plt.legend()
        plt.savefig(f"{state_plotpath}/{name}_inference_d={d}.png", dpi=200, bbox_inches="tight",)
        plt.close()

        print(f"{name}[{d}] posterior mean RMSE:", np.sqrt(np.mean((mean - truth) ** 2)))


####################################
#       plot replacement rate      #
####################################

replacement_rates = results["replacement_rates"]
replacement_posterior_slice = slice(args.burnin, args.burnin + args.samples)
posterior_replacement_rates = replacement_rates[replacement_posterior_slice]

# mean replacement rate across time at each Gibbs iteration
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


# replacement rate at each state time and Gibbs iteration
plt.figure(figsize=(10, 5))
image = plt.imshow(
    replacement_rates.T,
    aspect="auto",
    origin="lower",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    interpolation="nearest",
)
plt.axvline(args.burnin, linestyle="--", color="white")
plt.xlabel("Gibbs iteration")
plt.ylabel("State time")
colourbar = plt.colorbar(image)
colourbar.set_label("Replacement rate")
plt.tight_layout()
plt.savefig(f"{plotpath}/replacement_rate_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()


# posterior replacement rate across the state trajectory
posterior_replacement_rate = posterior_replacement_rates.mean(axis=0)

plt.figure()
plt.plot(posterior_replacement_rate, color="black")
plt.xlabel("State time")
plt.ylabel("Posterior mean replacement rate")
plt.ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig(f"{plotpath}/replacement_rate_by_time.png", dpi=200, bbox_inches="tight")
plt.close()

print("Posterior mean replacement rate:", posterior_replacement_rates.mean())

# import argparse
# import os

# import matplotlib.pyplot as plt
# import numpy as np

# from jax.scipy.linalg import solve_triangular


# parser = argparse.ArgumentParser()

# parser.add_argument("--T", type=int, default=100)
# parser.add_argument("--D", type=int, default=1)
# parser.add_argument("--steps", type=int, default=100)

# parser.add_argument("--burnin", type=int, default=500)
# parser.add_argument("--samples", type=int, default=500)

# parser.add_argument("--phi", type=float, default=0.1)
# parser.add_argument("--log-var", dest="log_var", type=float, default=0)

# parser.add_argument("--kernel", type=str, default="CSMC")

# parser.add_argument("--seed", type=int, default=1234)

# parser.add_argument("--conditional", action="store_true")
# parser.add_argument("--unconditional", dest="conditional", action="store_false")
# parser.set_defaults(conditional=True)

# parser.add_argument("--backward", action="store_true")
# parser.add_argument("--no-backward", dest="backward", action="store_false")
# parser.set_defaults(backward=True)

# parser.add_argument("--N", type=int, default=31)

# parser.add_argument("--i", type=int, default=0)
# parser.add_argument("--component", type=int, default=0)
# parser.add_argument("--n-paths", dest="n_paths", type=int, default=10)

# args = parser.parse_args()

# ########################
# #       Load data      #
# ########################

# experiment_name = "kernel={},D={},T={},phi={},log-var={},N={},samples={},burnin={},conditional={},seed={}"
# experiment_name = experiment_name.format(
#     args.kernel, 
#     args.D, 
#     args.T, 
#     args.phi, 
#     args.log_var, 
#     args.N,
#     args.samples, 
#     args.burnin, 
#     args.conditional, 
#     args.seed
# )

# dirpath = f"results/{experiment_name}"
# datapath = f"{dirpath}/data.npz"

# if not os.path.exists(datapath):
#     raise FileNotFoundError(f"Could not find saved data at {datapath}")

# plotpath = f"{dirpath}/plots"
# os.makedirs(plotpath, exist_ok=True)

# results = np.load(datapath, allow_pickle=True)
# print(f"Loaded results from: {dirpath}")

# # load posterior estimates and truth
# true_params = results["true_params"].item()
# param_hist = results["params"].item()
# sample_hist = results["trajectories"]
# true_xs = results["xs"]


# ##########################
# #       Plot params      #
# ##########################
# posterior_slice = slice(args.burnin + 1, args.burnin + args.samples + 1)

# # evaluate prior parameter inference
# true_H = true_params["H"]
# H_hist = param_hist["H"]
# H_mean = H_hist[posterior_slice].mean(axis=0)
# print("\nPosterior mean H:\n", H_mean)
# print("True H:\n", true_H)
# print("H absolute error:", np.abs(H_mean - true_H).sum())


# # plot precision matrices
# D = true_H.shape[0]
# true_prec_H = solve_triangular(true_H, np.eye(D))
# prec_H_mean = np.mean(np.linalg.inv(H_hist), axis=0)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# scale = max(np.max(np.abs(true_prec_H)), np.max(np.abs(prec_H_mean)))

# for ax, H, title in [(axes[0], true_prec_H, "True precision H"), (axes[1], prec_H_mean, "Posterior mean precision H")]:
#     image = ax.imshow(
#         H,
#         cmap="coolwarm",
#         vmin=-scale,
#         vmax=scale,
#         interpolation="nearest",
#     )

#     ax.set_title(title)
#     ax.set_xlabel("Column")
#     ax.set_ylabel("Row")
#     ax.set_xticks(np.arange(args.D))
#     ax.set_yticks(np.arange(args.D))

# # shared colour bar for both matrices
# colourbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
# colourbar.set_label("prec H value")

# # fig.subplots_adjust(left=0.07, right=0.88, bottom=0.10, top=0.90, wspace=0.25)
# plt.savefig(f"{plotpath}/prec_H_heatmaps.png", dpi=200, bbox_inches="tight")
# plt.close()


# # plot partial correlations
# def get_pcs(prec):
#     diags = np.diag(prec)

#     if np.any(diags <= 0):
#         raise ValueError("Precision matrix must have positive diagonal entries.")

#     pcs = -prec / np.sqrt(np.outer(diags, diags))
#     np.fill_diagonal(pcs, 1.0)

#     return pcs

# true_pcs = get_pcs(np.asarray(true_prec_H))
# posterior_pcs = get_pcs(prec_H_mean)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# scale = max(np.max(np.abs(true_pcs)), np.max(np.abs(posterior_pcs)))

# for ax, H, title in [(axes[0], true_pcs, "True precision pcs"), (axes[1], posterior_pcs, "Posterior mean pcs")]:
#     image = ax.imshow(
#         H,
#         cmap="coolwarm",
#         vmin=-scale,
#         vmax=scale,
#         interpolation="nearest",
#     )

#     ax.set_title(title)
#     ax.set_xlabel("Column")
#     ax.set_ylabel("Row")
#     ax.set_xticks(np.arange(args.D))
#     ax.set_yticks(np.arange(args.D))

# # shared colour bar for both matrices
# colourbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
# colourbar.set_label("pc value")

# # fig.subplots_adjust(left=0.07, right=0.88, bottom=0.10, top=0.90, wspace=0.25)
# plt.savefig(f"{plotpath}/pc_heatmaps.png", dpi=200, bbox_inches="tight")
# plt.close()

# # # plot loss over iterations
# # loss_history = results["loss_history"]

# # plt.figure()
# # plt.plot(loss_history)
# # plt.axvline(args.burnin, linestyle="--", color="black")
# # plt.xlabel("Iteration")
# # plt.ylabel("Loss")
# # plt.savefig(f"{plotpath}/loss_history.png", dpi=200, bbox_inches="tight")
# # plt.close()


# # H traces
# fig, axes = plt.subplots(args.D, args.D, figsize=(3 * args.D, 2.5 * args.D), squeeze=False)

# for j in range(args.D):
#     for k in range(args.D):
#         ax = axes[j, k]

#         if k > j:
#             ax.axis("off")
#             continue

#         ax.plot(H_hist[:, j, k])
#         ax.axhline(true_H[j, k], linestyle=":", color="red")
#         ax.axvline(args.burnin, linestyle="--", color="black")
#         ax.set_title(f"H[{j},{k}]")

# plt.tight_layout()
# plt.savefig(f"{plotpath}/H_traces.png", dpi=200, bbox_inches="tight")
# plt.close()


# # true and posterior mean H heatmaps
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # use one shared scale so colours are directly comparable
# scale = max(np.max(np.abs(true_H)), np.max(np.abs(H_mean)))

# for ax, H, title in [(axes[0], true_H, "True H"), (axes[1], H_mean, "Posterior mean H")]:
#     image = ax.imshow(
#         H,
#         cmap="coolwarm",
#         vmin=-scale,
#         vmax=scale,
#         interpolation="nearest",
#     )

#     ax.set_title(title)
#     ax.set_xlabel("Column")
#     ax.set_ylabel("Row")
#     ax.set_xticks(np.arange(args.D))
#     ax.set_yticks(np.arange(args.D))

# # shared colour bar for both matrices
# colourbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
# colourbar.set_label("H value")

# fig.subplots_adjust(left=0.07, right=0.88, bottom=0.10, top=0.90, wspace=0.25)
# plt.savefig(f"{plotpath}/H_heatmaps.png", dpi=200, bbox_inches="tight")
# plt.close()


# # tau trace
# tau_hist = param_hist["tau"]
# plt.figure()
# plt.plot(tau_hist)
# plt.title("Tau trace")
# plt.axvline(args.burnin, linestyle="--", color="black")
# plt.tight_layout()
# plt.savefig(f"{plotpath}/tau_trace.png", dpi=200, bbox_inches="tight")
# plt.close()


# # llambda traces
# llambda_hist = param_hist["llambda"]
# fig, axes = plt.subplots(args.D, args.D, figsize=(3 * args.D, 2.5 * args.D), squeeze=False)

# for j in range(args.D):
#     for k in range(args.D):
#         ax = axes[j, k]

#         if k >= j:
#             ax.axis("off")
#             continue

#         ax.plot(llambda_hist[:, j, k])
#         ax.axvline(args.burnin, linestyle="--", color="black")
#         ax.set_title(f"Lambda[{j},{k}]")

# plt.tight_layout()
# plt.savefig(f"{plotpath}/llambda_traces.png", dpi=200, bbox_inches="tight")
# plt.close()


# #################################
# #       plot trajectories       #
# #################################

# # state posterior estimates
# sample_zs, sample_etas = sample_hist
# true_zs, true_etas = true_xs

# posterior_zs = sample_zs[posterior_slice]
# posterior_etas = sample_etas[posterior_slice]

# true_etas = true_etas / 100
# posterior_etas = posterior_etas / 100 
# # d = min(args.component, args.D - 1)


# z_plotpath = f"{plotpath}/zs"
# eta_plotpath = f"{plotpath}/etas"

# os.makedirs(z_plotpath, exist_ok=True)
# os.makedirs(eta_plotpath, exist_ok=True)

# print(posterior_zs.shape)
# for d in range(args.D):
#     mean_z = posterior_zs[:, :, d].mean(axis=0)
#     mean_eta = posterior_etas[:, :, d].mean(axis=0)

#     for name, samples, mean, truth, state_plotpath in [
#         ("z", posterior_zs, mean_z, true_zs[:, d], z_plotpath),
#         ("eta", posterior_etas, mean_eta, true_etas[:, d], eta_plotpath),
#     ]:
#         plt.figure(figsize=(25, 5))
#         plt.plot(truth, label=f"true {name}", linestyle="--", color="blue")
#         plt.plot(mean, label="posterior mean", color="black")

#         for s in range(min(args.n_paths, samples.shape[0])):
#             plt.plot(samples[s, :, d], alpha=0.15, color="grey")

#         plt.xlabel("t")
#         plt.ylabel(f"{name}[{d}]")
#         plt.legend()
#         plt.savefig(f"{state_plotpath}/{name}_inference_d={d}.png", dpi=200, bbox_inches="tight",)
#         plt.close()

#         print(f"{name}[{d}] posterior mean RMSE:", np.sqrt(np.mean((mean - truth) ** 2)))


# ####################################
# #       plot replacement rate      #
# ####################################

# replacement_rates = results["replacement_rates"]
# replacement_posterior_slice = slice(args.burnin, args.burnin + args.samples)
# posterior_replacement_rates = replacement_rates[replacement_posterior_slice]

# # mean replacement rate across time at each Gibbs iteration
# mean_replacement_rate = replacement_rates.mean(axis=1)

# plt.figure()
# plt.plot(mean_replacement_rate)
# plt.axvline(args.burnin, linestyle="--", color="black")
# plt.xlabel("Gibbs iteration")
# plt.ylabel("Mean replacement rate")
# plt.ylim(0.0, 1.0)
# plt.tight_layout()
# plt.savefig(f"{plotpath}/replacement_rate_trace.png", dpi=200, bbox_inches="tight")
# plt.close()


# # replacement rate at each state time and Gibbs iteration
# plt.figure(figsize=(10, 5))
# image = plt.imshow(
#     replacement_rates.T,
#     aspect="auto",
#     origin="lower",
#     cmap="viridis",
#     vmin=0.0,
#     vmax=1.0,
#     interpolation="nearest",
# )
# plt.axvline(args.burnin, linestyle="--", color="white")
# plt.xlabel("Gibbs iteration")
# plt.ylabel("State time")
# colourbar = plt.colorbar(image)
# colourbar.set_label("Replacement rate")
# plt.tight_layout()
# plt.savefig(f"{plotpath}/replacement_rate_heatmap.png", dpi=200, bbox_inches="tight")
# plt.close()


# # posterior replacement rate across the state trajectory
# posterior_replacement_rate = posterior_replacement_rates.mean(axis=0)

# plt.figure()
# plt.plot(posterior_replacement_rate, color="black")
# plt.xlabel("State time")
# plt.ylabel("Posterior mean replacement rate")
# plt.ylim(0.0, 1.0)
# plt.tight_layout()
# plt.savefig(f"{plotpath}/replacement_rate_by_time.png", dpi=200, bbox_inches="tight")
# plt.close()

# print("Posterior mean replacement rate:", posterior_replacement_rates.mean())