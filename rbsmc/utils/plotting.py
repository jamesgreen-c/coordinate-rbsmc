import matplotlib.pyplot as plt
import numpy as np


def plot_rr_v_delta(
        rr: np.array, 
        delta: np.array,
        plotdir: str
    ):
    """
    Plot the replacement rate versus the delta used

    rr: (K, M, T)  The mean replacement rate across samples for experiment K, chain M, time T 
    delta: (K, T)   The (adapted) delta which can vary over T
    """
    K, M, T = rr.shape

    fig, ax = plt.subplots(K, 1, figsize=(15, 5))
    
    for k in range(K):
        ax.plot(rr[k].T, alpha=0.5, label="Replacement rate")
        plt.legend()
        ax.twinx().semilogy(delta[k].T, alpha=0.5, label="Delta", color="black", ls="--")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{plotdir}/replace_rate_and_delta.png")
    plt.close(fig)


def plot_xs(
        init_xs: np.array,
        true_xs: np.array,
        means: np.array,
        std: np.array,
        plotdir: str,
        components: list | None = None
    ):
    """
    Plot the true latent state against the initial xs used (e.g BPF output) against the mean +- std sampled xs
    Loop over D first as it is likely much higher than K

    init_xs: (K, T, D)
    true_xs: (K, T, D)
    means: (K, M, T, D)
    std: (K, M, T, D)
    components: a list of indices of components to plot, if None plot all
    """
    K, M, T, D = means.shape

    components = components if components is not None else range(D)

    for d in components:
        fig, ax = plt.subplots(K, 1, figsize=(15, K*5), squeeze=False)
        for k in range(K):
            ax[k, 0].plot(init_xs[k, :, d], label="Init", color="green")
            for m in range(M):
                ax[k, 0].plot(means[k, m, :, d], alpha=0.5, label=f"Mean {m}")
                ax[k, 0].fill_between(np.arange(T),
                                    means[k, m, :, d] - 2 * std[k, m, :, d],
                                    means[k, m, :, d] + 2 * std[k, m, :, d],
                                    alpha=0.3)
            ax[k, 0].plot(true_xs[k, :, d], label="True", color="black", linestyle="dashed")
            ax[k, 0].legend()
        plt.tight_layout()
        plt.savefig(f"{plotdir}/state_{d}.png")
        plt.close(fig)


def plot_ess(
        ess: np.array,
        plotdir: str
    ):
    """
    Plot the ESS per dimension of the latent state

    ess: (K, T, D)
    """
    K, T, D = ess.shape

    fig, ax = plt.subplots(K, 1, figsize=(15, K*5), squeeze=False)
    for k in range(K):
        for d in range(D):
            ax[k, 0].plot(ess[k, :, d], label=f"Dim {d}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plotdir}/ess.png")
    plt.close(fig)


def plot_traces(
        traces: np.array,
        plotdir: str
    ):
    """
    
    traces (K, N, M, 3)  trace for experiment K, over N samples, for M chains, at 3 different T, for dim zero
    """
    K, N, M, D = traces.shape    

    fig, ax = plt.subplots(K, D, figsize=(D*15, 5*K), squeeze=False)
    labels = ["t=0", r"t=$\frac{T}{2}$", "t=T"]
    for k in range(K):
        for d in range(D):
            ax[k, d].plot(traces[k, ..., d], label=labels[d])
            ax[k, d].legend()
            ax[k, d].set_xlabel("sample")
    plt.tight_layout()
    plt.savefig(f"{plotdir}/trace_plots.png")
    plt.close()


def plot_square_error(
    init_xs: np.ndarray,
    true_xs: np.ndarray,
    means: np.ndarray,
    plotdir: str,
):
    """
    Compare squared error from true_xs for the initialisation path and sampled means.

    init_xs: (K, T, D)
    true_xs: (K, T, D)
    means:   (K, M, T, D)
    """
    K, M, T, D = means.shape

    # Squared errors
    init_se = (init_xs - true_xs) ** 2            # (K, T, D)
    means_se = (means - true_xs[:, None, :, :]) ** 2  # (K, M, T, D) 

    # Sum Squared Error
    init_sse = init_se.sum(axis=1)        # (K, D)
    means_sse = means_se.sum(axis=2)      # (K, M, D)

    x = np.arange(D)
    n_bars = M + 1                        # init + M chains
    group_width = 0.8
    bar_w = group_width / n_bars
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_w  # centered offsets

    fig, ax = plt.subplots(K, 1, figsize=(max(10, 0.8 * D), 4 * K), squeeze=False)

    for k in range(K):

        # init bars
        ax[k, 0].bar(x + offsets[0], init_sse[k], width=bar_w, label="Init")

        # chain bars
        for m in range(M):
            ax[k, 0].bar(x + offsets[m + 1], means_sse[k, m], width=bar_w, label=f"Chain {m}")

        ax[k, 0].set_title(f"Squared error (SSE over time), k={k}")
        ax[k, 0].set_xlabel("Dimension d")
        ax[k, 0].set_ylabel("SSE")
        ax[k, 0].set_xticks(x)
        ax[k, 0].set_xticklabels([str(d) for d in range(D)])
        ax[k, 0].legend(ncol=min(4, n_bars))

    fig.tight_layout()
    fig.savefig(f"{plotdir}/squared_errors.png", dpi=200)
    plt.close(fig)


def plot_mae(
    init_xs: np.ndarray,
    true_xs: np.ndarray,
    means: np.ndarray,
    plotdir: str,
):
    """
    Compare mean absolute error from true_xs for the initialisation path and sampled means.

    init_xs: (K, T, D)
    true_xs: (K, T, D)
    means:   (K, M, T, D)
    """
    K, M, T, D = means.shape

    # Absolute errors
    init_se = np.abs(init_xs - true_xs)                 # (K, T, D)
    means_se = np.abs(means - true_xs[:, None, :, :])   # (K, M, T, D) 

    # Mean Absolute Error
    init_mae = init_se.mean(axis=1)        # (K, D)
    means_mae = means_se.mean(axis=2)      # (K, M, D)

    x = np.arange(D)
    n_bars = M + 1                         # init + M chains
    group_width = 0.8
    bar_w = group_width / n_bars
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_w  # centered offsets

    fig, ax = plt.subplots(K, 1, figsize=(max(10, 0.8 * D), 4 * K), squeeze=False)

    for k in range(K):

        # init bars
        ax[k, 0].bar(x + offsets[0], init_mae[k], width=bar_w, label="Init")

        # chain bars
        for m in range(M):
            ax[k, 0].bar(x + offsets[m + 1], means_mae[k, m], width=bar_w, label=f"Chain {m}")

        ax[k, 0].set_title(f"Mean absolute error (MAE over time), k={k}")
        ax[k, 0].set_xlabel("Dimension d")
        ax[k, 0].set_ylabel("MAE")
        ax[k, 0].set_xticks(x)
        ax[k, 0].set_xticklabels([str(d) for d in range(D)])
        ax[k, 0].legend(ncol=min(4, n_bars))

    fig.tight_layout()
    fig.savefig(f"{plotdir}/mae.png", dpi=200)
    plt.close(fig)