#!/usr/bin/env python3
r"""Compare saved corporate-bond particle Gibbs experiments.

Run from the SAME repository/environment as experiment.py: dataset objects in
the NPZ files need their original Python classes (and JAX) to be unpickled.
This script does not regenerate data or rerun inference.

Examples (place this file beside distribute.py, or supply --root):
    python analysis_compare.py --root experiments/synth/low_frequency
    python analysis_compare.py --root experiments/synth \
        --results-dir low=experiments/synth/low_frequency/results \
        --results-dir high=experiments/synth/high_frequency/results
    python analysis_compare.py --root experiments/synth --dims 3 10 15 20 50 \
        --N 31 --samples 500 --burnin 1000 --plot-D 20 --bond 0

Dependencies: numpy, matplotlib, and scipy for rank-normalised diagnostics.
ESS and R-hat are implemented locally; no Bayesian analysis package is needed.
Use --no-ess to omit ESS/R-hat, or --ess-method mean for raw split-chain ESS.

Output defaults to ROOT/plots. Each matched inference configuration gets its
own subdirectory containing:
    H_error_by_D.{png,pdf}            relative Frobenius error and entrywise MAE
    replacement_and_ESS_by_D.*        replacement plus diagonal/off-diagonal ESS
    eta_paths_D=20_seed=1234.*        one aligned panel per available method
    H_heatmaps_D=20_seed=1234.*        truth and posterior mean covariance
    partial_correlations_D=20_seed=1234.*   draw-wise posterior means
    selected_traces_D=20_seed=1234.*  variance; weak/medium/strong nonzero PCs and a zero PC
    selected_ACF_D=20_seed=1234.*     lags in original Gibbs iterations
    nonzero_pc_traces_*.pdf          multipage traces of EVERY truly nonzero PC
    zero_pc_traces_*.pdf             reproducibly selected truly zero PCs
    worst_mixing_pc_traces_*.pdf     union of worst-mixing PCs across methods
    *_page=NNN.{png,svg}             trace pages in requested image formats
    partial_correlation_ESS_by_D.*   separate summaries for zero/nonzero PCs
CSV tables, a JSON manifest, and analysis_notes.txt are saved in ROOT/plots.

Important interpretation:
* H is transformed back with diag(scales) @ H @ diag(scales).
* Initialisation is excluded; burn-in uses the saved config, not array offsets.
* Reference paths are chronological, but only the latest saved_paths survive.
* ESS is for retained draws. ESS/draw cannot recover unthinned efficiency.
* Runtime is not saved by the supplied experiment.py; ESS/second is unavailable.
* Saved replacement_rates are rolling means of aux['replaced']. This file cannot
  determine whether each kernel reports replacement of active coordinates or
  reconstructed full states. It labels the diagnostic as reported replacement.
* experiment.py fixes its data-generation key, ignores --M, and varies the
  sampler seed. Repeated seeds usually describe sampler variability on one data
  set, not independently simulated data sets. Dataset fingerprints distinguish
  these cases. Replicate bands are descriptive 10th--90th percentiles, not CIs.
* Nonlinear posterior summaries (partial correlations) are transformed per draw.
* True zero PCs have abs(value)<=--pc-zero-tol (default 1e-8). Sparsity of H
  itself does not imply sparse PCs: these are determined by inverse(H).
* Local bulk ESS uses split chains, pooled rank normalisation, FFT covariance,
  and Geyer's initial positive/monotone paired autocorrelation sequence. Mean
  ESS uses the same calculation on raw split chains. Odd-length chains lose
  their middle draw when split. Rank R-hat also includes the folded diagnostic.
* Stuck/nonfinite entries have NaN ESS and are counted in exported diagnostics.
* No new observation-frequency regime or augmentation experiment is fabricated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


METHODS = ("RB_CSMC", "CSMC", "GUEANT")
LABELS = {"RB_CSMC": "Coordinate RBPF", "CSMC": "Bootstrap PF", "GUEANT": "Guided PF"}
MODES = {"RB_CSMC": "reduced", "CSMC": "ffbsi", "GUEANT": "ffbsi"}
COLOURS = {"RB_CSMC": "#0072B2", "CSMC": "#D55E00", "GUEANT": "#009E73"}
ALIASES = {"0": "CSMC", "1": "RB_CSMC", "2": "GUEANT"}


@dataclass
class Run:
    path: Path
    meta: dict[str, Any]
    config: dict[str, Any]
    source_label: str | None
    cohort: tuple
    regime: str
    dataset_id: str
    truth: np.ndarray
    H_mean: np.ndarray
    pcs_truth: np.ndarray
    pcs_mean: np.ndarray
    metrics: dict[str, Any]
    ess: np.ndarray
    pcs_ess: np.ndarray
    rows: np.ndarray
    cols: np.ndarray


class Analysis:
    def __init__(self, args):
        self.args = args
        self.notes: list[str] = []
        self.outputs: list[str] = []
        self.ess_enabled = not args.no_ess

    def note(self, message):
        if message not in self.notes:
            self.notes.append(message)
            print(f"NOTE: {message}", file=sys.stderr)

    def save(self, fig, directory, stem, formats=None):
        directory.mkdir(parents=True, exist_ok=True)
        for fmt in self.args.formats if formats is None else formats:
            path = directory / f"{stem}.{fmt}"
            fig.savefig(path, dpi=self.args.dpi, bbox_inches="tight")
            self.outputs.append(str(path))
        plt.close(fig)


def unpack(value):
    if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        return value.item()
    return value


def mapping(value):
    value = unpack(value)
    if isinstance(value, dict):
        return value
    if hasattr(value, "_asdict"):
        return value._asdict()
    return vars(value)


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def name_metadata(path):
    values = dict(re.findall(r"(?:^|,)([^=,]+)=([^,]+)", path.parent.name))
    for key in ("D", "T", "steps", "N", "samples", "burnin", "seed", "thin"):
        if key in values:
            values[key] = int(values[key])
    kernel = str(values.get("kernel", ""))
    values["kernel"] = ALIASES.get(kernel, kernel)
    values["backward"] = values.get("backward", values.get("backward-mode", "ffbsi"))
    return values


def iteration_coordinates(config, history_size):
    """Match training.py: slot 0 is initialisation; itr%thin==0 is retained."""
    burnin, samples, thin = (int(config[k]) for k in ("burnin", "samples", "thin"))
    if burnin < 0 or samples < 0 or thin < 1:
        raise ValueError("Invalid burnin/samples/thin in saved config.")
    retained = np.arange(0, burnin + samples, thin, dtype=int)
    params = np.concatenate(([-1], retained))
    if len(params) != history_size:
        raise ValueError(
            f"H history length {history_size} does not match config length {len(params)}."
        )
    if not np.any(params >= burnin):
        raise ValueError("No retained H samples after burn-in.")
    return params, retained


def original_H(npz):
    params = mapping(npz["params"])
    history = np.asarray(params["H"], dtype=float)
    scales = np.asarray(npz["standardisation_scales"], dtype=float).reshape(-1)
    if history.ndim != 3 or history.shape[1:] != (len(scales), len(scales)):
        raise ValueError(f"Expected H history (draw,D,D); got {history.shape}.")
    if np.any(scales <= 0) or not np.all(np.isfinite(scales)):
        raise ValueError("Standardisation scales must be finite and positive.")
    return history * scales[None, :, None] * scales[None, None, :]


def partial_correlations(covariances):
    precision = np.linalg.inv(covariances)
    diagonal = np.diagonal(precision, axis1=-2, axis2=-1)
    if np.any(diagonal <= 0):
        raise ValueError("Precision has a nonpositive diagonal.")
    result = -precision / np.sqrt(diagonal[..., :, None] * diagonal[..., None, :])
    idx = np.arange(result.shape[-1])
    result[..., idx, idx] = 1.0
    return result


def mean_partial_correlations(history):
    # Limit temporary matrix inverses, especially for larger D.
    total = np.zeros(history.shape[1:], dtype=float)
    for start in range(0, len(history), 32):
        total += partial_correlations(history[start:start + 32]).sum(axis=0)
    return total / len(history)


def dataset_fingerprint(dataset, truth, npz):
    """Hash observations, generative parameters and scaling; never equate seeds to data."""
    digest = hashlib.sha256()
    data = field(dataset, "data")
    if data is None:
        raise ValueError("Saved dataset has no data attribute.")
    for value in (*data, *[truth[k] for k in sorted(truth)],
                  npz["standardisation_means"], npz["standardisation_scales"]):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    # Fixed fitted nuisance parameters affect the target as well.
    if "estimated_params" in npz.files:
        for key, value in sorted(mapping(npz["estimated_params"]).items()):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(np.asarray(value)).tobytes())
    for name in ("dts", "times"):
        value = field(dataset, name)
        if value is not None:
            digest.update(np.ascontiguousarray(np.asarray(value)).tobytes())
    return digest.hexdigest()[:16]


def numerical_summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.median(values)), float(np.quantile(values, 0.1))


def split_chains(chains):
    """Input (chain,draw,entry); concatenate equal first/last halves."""
    chains = np.asarray(chains, dtype=float)
    half = chains.shape[1] // 2
    return np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)


def rank_normalise(chains):
    """Pooled midranks and Blom's normal-score transform, entry by entry."""
    from scipy.special import ndtri
    from scipy.stats import rankdata

    flat = chains.reshape(-1, chains.shape[-1])
    ranks = rankdata(flat, method="average", axis=0)
    scores = ndtri((ranks - 0.375) / (len(flat) + 0.25))
    return scores.reshape(chains.shape)


def fft_autocovariance(chains):
    """Biased autocovariance, divisor n, with zero padding to avoid wraparound."""
    chains = np.asarray(chains, dtype=float)
    n = chains.shape[1]
    centred = chains - chains.mean(axis=1, keepdims=True)
    length = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=length, axis=1)
    return np.fft.irfft(spectrum * spectrum.conj(), n=length, axis=1)[:, :n].real / n


def geyer_ess(chains):
    """ESS for already split/transformed chains, vectorised over entries.

    rho[t] = 1 - (W - mean_chain_autocov[t]) / var_plus, with rho[0]=1.
    Pair rho[0]+rho[1], rho[2]+rho[3], ...; truncate at the first nonpositive
    pair and monotonise the retained positive sequence. A final positive even
    lag reduces variance for antithetic chains. The finite-sample lower bound
    on integrated autocorrelation time allows ESS > draws for negative ACF.
    """
    m, n, count = chains.shape
    out = np.full(count, np.nan)
    if n < 4:
        return out
    cov = fft_autocovariance(chains)
    within = (cov[:, 0] * n / (n - 1)).mean(axis=0)
    between = np.var(chains.mean(axis=1), axis=0, ddof=1) if m > 1 else np.zeros(count)
    variance = (n - 1) / n * within + between
    valid = np.isfinite(variance) & (variance > 0)
    rho = np.zeros((n, count))
    rho[:, valid] = 1 - (within[valid][None, :] - cov.mean(axis=0)[:, valid]) / variance[valid][None, :]
    rho[0] = 1.0
    # Work entrywise only on the short truncation sequence, not on raw samples.
    for j in np.flatnonzero(valid):
        positive = []
        pair_sum = 1.0 + rho[1, j]
        final_even = 1.0
        # Leave the terminal evaluated pair as the single-even-lag correction.
        # Do not use the last two noisy lags to complete another pair.
        for even_lag in range(2, n - 2, 2):
            if pair_sum <= 0:
                break
            positive.append(pair_sum)
            final_even = rho[even_lag, j]
            pair_sum = final_even + rho[even_lag + 1, j]
        monotone = np.minimum.accumulate(positive) if positive else np.empty(0)
        tau = -1 + 2 * monotone.sum() + max(0.0, final_even)
        tau = max(float(tau), 1 / np.log10(m * n))
        out[j] = m * n / tau
    return out


def local_ess(chains, method="bulk"):
    """Entrywise split ESS; shapes (chain,draw,entry) or (draw,entry)."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim == 2:
        chains = chains[None, ...]
    if chains.ndim != 3 or method not in ("bulk", "mean"):
        raise ValueError("ESS requires (chain,draw,entry) and method bulk or mean.")
    out = np.full(chains.shape[-1], np.nan)
    if chains.shape[1] < 8:
        return out
    valid = np.all(np.isfinite(chains), axis=(0, 1)) & (np.ptp(chains, axis=(0, 1)) > 0)
    if np.any(valid):
        values = split_chains(chains[:, :, valid])
        if method == "bulk":
            values = rank_normalise(values)
        out[valid] = geyer_ess(values)
    return out


def local_rank_rhat(chains):
    """Maximum of rank-normalised and folded split R-hat for each entry."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3 or chains.shape[0] < 2:
        raise ValueError("R-hat requires at least two independent chains.")
    out = np.full(chains.shape[-1], np.nan)
    if chains.shape[1] < 8:
        return out
    valid = np.all(np.isfinite(chains), axis=(0, 1)) & (np.ptp(chains, axis=(0, 1)) > 0)
    if not np.any(valid):
        return out
    values = split_chains(chains[:, :, valid])

    def rhat(data):
        n = data.shape[1]
        chain_variances = np.var(data, axis=1, ddof=1)
        chain_variances[np.ptp(data, axis=1) == 0] = 0.0
        within = chain_variances.mean(axis=0)
        between_over_n = np.var(data.mean(axis=1), axis=0, ddof=1)
        variance = (n - 1) / n * within + between_over_n
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(variance / within)

    ranked = rhat(rank_normalise(values))
    folded = np.abs(values - np.median(values, axis=(0, 1), keepdims=True))
    # All equal folded values carry no scale disagreement (e.g. two-point data).
    folded_rhat = rhat(rank_normalise(folded))
    folded_rhat[np.ptp(folded, axis=(0, 1)) == 0] = 1.0
    out[valid] = np.maximum(ranked, folded_rhat)
    return out


def estimate_ess(analysis, entries):
    if not analysis.ess_enabled:
        return np.full(entries.shape[-1], np.nan)
    return local_ess(entries, method=analysis.args.ess_method)


def pc_history(history):
    """Compute per-draw PCs using bounded-size batches of matrix inverses."""
    output = np.empty_like(history, dtype=float)
    for start in range(0, len(history), 32):
        output[start:start + 32] = partial_correlations(history[start:start + 32])
    return output


def analyse_run(path, source_label, meta, analysis):
    with np.load(path, allow_pickle=True) as npz:
        config = mapping(npz["config"]) if "config" in npz.files else {}
        config = {**{k: meta.get(k, default) for k, default in
                    (("burnin", 500), ("samples", 500), ("thin", 1))}, **config}
        H = original_H(npz)
        coords, retained = iteration_coordinates(config, len(H))
        mask = coords >= int(config["burnin"])
        posterior = H[mask]
        D = H.shape[1]
        if D != meta["D"] or not np.all(np.isfinite(H)):
            raise ValueError("Non-finite H samples or dimension mismatch.")
        np.linalg.cholesky(posterior)  # validate SPD rather than plotting invalid covariances
        truth_params = mapping(npz["true_params"])
        truth = np.asarray(truth_params["H"], dtype=float)
        if truth.shape != (D, D):
            raise ValueError("True H has the wrong shape.")
        np.linalg.cholesky(truth)
        dataset = unpack(npz["dataset"])
        dataset_id = dataset_fingerprint(dataset, truth_params, npz)
        mean = posterior.mean(axis=0)
        pcs_true = partial_correlations(truth)
        posterior_pcs = pc_history(posterior)
        pcs_mean = posterior_pcs.mean(axis=0)
        rows, cols = np.triu_indices(D)
        ess = estimate_ess(analysis, posterior[:, rows, cols])
        pcs_ess = estimate_ess(analysis, posterior_pcs[:, rows, cols])
        relative = ess / len(posterior)
        diagonal = rows == cols
        off_diagonal = ~diagonal
        delta = mean - truth
        metrics = {
            "H_mae": float(np.mean(np.abs(delta))),
            "H_relative_frobenius": float(np.linalg.norm(delta) / np.linalg.norm(truth)),
            "H_diagonal_mae": float(np.mean(np.abs(np.diag(delta)))),
            "H_off_diagonal_mae": float(np.mean(np.abs(delta[~np.eye(D, dtype=bool)])))
                if D > 1 else float("nan"),
            "partial_correlation_mae": float(np.mean(np.abs((pcs_mean - pcs_true)[rows[off_diagonal], cols[off_diagonal]])))
                if D > 1 else float("nan"),
            "retained_posterior_draws": len(posterior),
            "thin": int(config["thin"]),
            "dataset_id": dataset_id,
        }
        for name, selector in (("diagonal", diagonal), ("off_diagonal", off_diagonal)):
            median, p10 = numerical_summary(relative[selector])
            metrics[f"{name}_relative_ess_median"] = median
            metrics[f"{name}_relative_ess_p10"] = p10
            metrics[f"{name}_valid_ess_entries"] = int(np.isfinite(ess[selector]).sum())
        pc_nonzero = off_diagonal & (np.abs(pcs_true[rows, cols]) > analysis.args.pc_zero_tol)
        pc_zero = off_diagonal & ~pc_nonzero
        for name, selector in (("nonzero", pc_nonzero), ("zero", pc_zero)):
            median, p10 = numerical_summary(pcs_ess[selector] / len(posterior))
            metrics[f"pc_{name}_relative_ess_median"] = median
            metrics[f"pc_{name}_relative_ess_p10"] = p10
            metrics[f"pc_{name}_entries"] = int(selector.sum())
            metrics[f"pc_{name}_invalid_ess_entries"] = int((~np.isfinite(pcs_ess[selector])).sum())
        metrics["ess_method"] = analysis.args.ess_method
        metrics["pc_zero_tolerance"] = analysis.args.pc_zero_tol
        replacement = np.asarray(npz["replacement_rates"], dtype=float)
        if replacement.ndim != 2 or len(replacement) != len(retained):
            raise ValueError("Replacement history disagrees with saved config.")
        if not np.all(np.isfinite(replacement)) or np.any((replacement < 0) | (replacement > 1)):
            raise ValueError("Replacement rates must lie in [0,1].")
        # Avoid rolling windows containing pre-burn-in iterations where possible.
        width = int(config.get("replacement_rate_window", 100))
        clean_mask = retained >= int(config["burnin"]) + width - 1
        if not np.any(clean_mask):
            clean_mask = retained >= int(config["burnin"])
            analysis.note(f"{path.parent.name}: replacement windows overlap burn-in.")
        metrics["replacement_rate"] = float(replacement[clean_mask].mean())
        metrics["replacement_windows"] = int(clean_mask.sum())
        if len(posterior) < 50:
            analysis.note(f"{path.parent.name}: only {len(posterior)} retained posterior draws; ESS and credible bands are exploratory.")
        if int(config["thin"]) > 1:
            analysis.note(f"Thinning={config['thin']}: ESS is based on retained draws; unthinned autocorrelation cannot be recovered.")
        horizon = float(meta["T"]) - 1.0
        rate = meta["steps"] / (horizon * D) if horizon > 0 else float("nan")
        metrics["nominal_observations_per_bond_per_day"] = rate
        # Separate budgets so incomparable result configurations are not silently pooled.
        cohort = (meta["T"], meta["N"], int(config["burnin"]), int(config["samples"]),
                  int(config["thin"]), meta.get("full-inference", "unspecified"),
                  meta.get("phi", "unspecified"), meta.get("conditional", "unspecified"))
        return Run(path, meta, config, source_label, cohort, "", dataset_id, truth,
                   mean, pcs_true, pcs_mean, metrics, ess, pcs_ess, rows, cols)


def assign_regimes(runs):
    """Cluster nominal rates, allowing the integer rounding in distribute.py."""
    clusters: list[tuple[float, str]] = []
    for run in sorted(runs, key=lambda r: r.metrics["nominal_observations_per_bond_per_day"]):
        rate = run.metrics["nominal_observations_per_bond_per_day"]
        if run.source_label:
            base = run.source_label
        else:
            base = ""
        found = next((label for centre, label in clusters
                      if np.isclose(rate, centre, rtol=0.01, atol=1e-5)), None)
        if found is None:
            if np.isclose(rate, 1 / 3, rtol=0.01):
                found = "One observation per bond every 3 days"
            elif np.isclose(rate, 5, rtol=0.01):
                found = "Five observations per bond per day"
            else:
                found = f"{rate:.3g} observations per bond per day"
            clusters.append((rate, found))
        run.regime = f"{base}: {found}" if base else found


def slug(text):
    return re.sub(r"[^A-Za-z0-9_=.-]+", "_", text).strip("_")


def cohort_name(cohort):
    T, N, burnin, samples, thin, full, phi, conditional = cohort
    base = f"T={T},N={N},burnin={burnin},samples={samples},thin={thin}"
    for name, value in (("full", full), ("phi", phi), ("conditional", conditional)):
        if value != "unspecified":
            base += f",{name}={value}"
    return slug(base)


def metric_curve(ax, runs, metric, kernel, label=None, linestyle="-", band=True):
    subset = [r for r in runs if r.meta["kernel"] == kernel]
    dimensions = sorted({r.meta["D"] for r in subset})
    centre, low, high = [], [], []
    for D in dimensions:
        values = np.array([r.metrics[metric] for r in subset if r.meta["D"] == D])
        values = values[np.isfinite(values)]
        centre.append(np.median(values) if len(values) else np.nan)
        low.append(np.quantile(values, 0.1) if len(values) > 1 else np.nan)
        high.append(np.quantile(values, 0.9) if len(values) > 1 else np.nan)
    ax.plot(dimensions, centre, color=COLOURS[kernel], marker="o", markersize=4,
            linestyle=linestyle, linewidth=1.5, label=label or LABELS[kernel])
    if band:
        ax.fill_between(dimensions, low, high, color=COLOURS[kernel], alpha=0.13)
    ax.set_xticks(sorted({r.meta["D"] for r in runs}))
    ax.set_xlabel("Number of bonds, D")
    ax.grid(alpha=0.2)


def plot_scaling(runs, analysis, directory):
    regimes = list(dict.fromkeys(r.regime for r in runs))
    fig, axes = plt.subplots(2, len(regimes), figsize=(5 * len(regimes), 7), squeeze=False,
                             layout="constrained")
    for col, regime in enumerate(regimes):
        subset = [r for r in runs if r.regime == regime]
        for row, metric, ylabel in ((0, "H_relative_frobenius", "Relative Frobenius error in H"),
                                    (1, "H_mae", "Entrywise MAE in H")):
            ax = axes[row, col]
            for kernel in METHODS:
                metric_curve(ax, subset, metric, kernel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
        axes[0, col].set_title(regime, fontsize=10, wrap=True)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Covariance recovery: medians and 10–90% replicate ranges")
    analysis.save(fig, directory, "H_error_by_D")

    nrows = 3 if analysis.ess_enabled else 1
    fig, axes = plt.subplots(nrows, len(regimes), figsize=(5 * len(regimes), 2.7 * nrows),
                             squeeze=False, layout="constrained")
    for col, regime in enumerate(regimes):
        subset = [r for r in runs if r.regime == regime]
        for kernel in METHODS:
            metric_curve(axes[0, col], subset, "replacement_rate", kernel)
        axes[0, col].set_ylabel("Mean reported replacement rate")
        axes[0, col].set_ylim(0, 1)
        axes[0, col].set_title(regime, fontsize=10, wrap=True)
        if nrows == 3:
            for row, entry_class in ((1, "diagonal"), (2, "off_diagonal")):
                for kernel in METHODS:
                    metric_curve(axes[row, col], subset, f"{entry_class}_relative_ess_median", kernel)
                    metric_curve(axes[row, col], subset, f"{entry_class}_relative_ess_p10", kernel,
                                 linestyle="--", band=False, label="_nolegend_")
                axes[row, col].set_ylabel(f"{'Diagonal' if row == 1 else 'Off-diagonal'} {analysis.args.ess_method} ESS / retained draws")
                axes[row, col].set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=9)
    if nrows == 3:
        axes[1, 0].legend(handles=[Line2D([], [], color="0.25", label="Median entrywise ESS"),
                                  Line2D([], [], color="0.25", linestyle="--", label="Lower decile")], fontsize=9)
    fig.suptitle("Trajectory movement and covariance mixing (retained histories)")
    analysis.save(fig, directory, "replacement_and_ESS_by_D")
    if analysis.ess_enabled:
        fig, axes = plt.subplots(2, len(regimes), figsize=(5 * len(regimes), 6),
                                 squeeze=False, layout="constrained")
        for col, regime in enumerate(regimes):
            subset = [r for r in runs if r.regime == regime]
            for row, kind in enumerate(("nonzero", "zero")):
                for kernel in METHODS:
                    metric_curve(axes[row, col], subset, f"pc_{kind}_relative_ess_median", kernel)
                    metric_curve(axes[row, col], subset, f"pc_{kind}_relative_ess_p10", kernel,
                                 linestyle="--", band=False, label="_nolegend_")
                axes[row, col].set_ylabel(f"Truly {kind} PCs\n{analysis.args.ess_method} ESS / retained draws")
                axes[row, col].set_ylim(bottom=0)
                if not any(r.metrics[f"pc_{kind}_entries"] for r in subset):
                    axes[row, col].text(0.5, 0.5, f"No truly {kind} pairs", ha="center",
                                        transform=axes[row, col].transAxes)
            axes[0, col].set_title(regime, fontsize=10, wrap=True)
        axes[0, 0].legend(fontsize=8)
        axes[1, 0].legend(handles=[Line2D([], [], color="0.25", label="Median entrywise ESS"),
                                  Line2D([], [], color="0.25", linestyle="--", label="Lower decile")], fontsize=8)
        fig.suptitle(f"Partial-correlation mixing; true-zero tolerance {analysis.args.pc_zero_tol:g}")
        analysis.save(fig, directory, "partial_correlation_ESS_by_D")


def representative_sets(runs, analysis):
    """Require the same seed AND data across the displayed method panels."""
    for regime in dict.fromkeys(r.regime for r in runs):
        candidates = [r for r in runs if r.regime == regime and r.meta["D"] == analysis.args.plot_D]
        if not candidates:
            analysis.note(f"No D={analysis.args.plot_D} result in {regime}; representative panels skipped.")
            continue
        buckets = defaultdict(list)
        for run in candidates:
            buckets[(run.meta["seed"], run.dataset_id, run.meta["steps"])].append(run)
        keys = sorted(buckets, key=lambda key: (
            key[0] != analysis.args.plot_seed,
            -len({r.meta["kernel"] for r in buckets[key]}), key[0], key[1]))
        key = keys[0]
        chosen = buckets[key]
        if key[0] != analysis.args.plot_seed:
            analysis.note(f"Representative seed {analysis.args.plot_seed} unavailable in {regime}; using seed {key[0]}.")
        chosen = sorted(chosen, key=lambda r: METHODS.index(r.meta["kernel"]))
        if len(chosen) < 3:
            analysis.note(f"{regime}, D={analysis.args.plot_D}, seed={key[0]}: only {len(chosen)} matched methods; missing panels labelled.")
        yield regime, chosen


def plot_heatmaps(runs, analysis, directory, suffix):
    lookup = {r.meta["kernel"]: r for r in runs}
    first = runs[0]
    for kind, truth, getter, cmap, fixed in (
        ("H_heatmaps", first.truth, lambda r: r.H_mean, "coolwarm", False),
        ("partial_correlations", first.pcs_truth, lambda r: r.pcs_mean, "coolwarm", True),
    ):
        values = [truth] + [getter(lookup[k]) if k in lookup else None for k in METHODS]
        vmax = 1.0 if fixed else max(float(np.max(np.abs(v))) for v in values if v is not None)
        vmax = max(vmax, np.finfo(float).eps)
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), layout="constrained")
        image = None
        for ax, value, title in zip(axes, values, ["Truth"] + [LABELS[k] for k in METHODS]):
            ax.set_title(title, fontsize=10)
            if value is None:
                ax.text(0.5, 0.5, "Result unavailable", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            image = ax.imshow(value, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
            ticks = np.unique(np.linspace(0, value.shape[0] - 1, min(5, value.shape[0]), dtype=int))
            ax.set_xticks(ticks, ticks + 1)
            ax.set_yticks(ticks, ticks + 1)
            ax.set_xlabel("Bond")
        axes[0].set_ylabel("Bond")
        fig.colorbar(image, ax=list(axes), shrink=0.78,
                     label="Partial correlation" if fixed else "Covariance (original units)")
        fig.suptitle(f"{'Partial-correlation' if fixed else 'Covariance'} posterior means; {suffix.replace('_', ', ')}")
        analysis.save(fig, directory, f"{kind}_{suffix}")


def observation_times(dataset, count):
    for key in ("times", "ts", "timestamps"):
        value = field(dataset, key)
        if value is not None:
            value = np.asarray(value, dtype=float)
            if value.shape == (count,) and np.all(np.isfinite(value)) and np.all(np.diff(value) >= 0):
                return value, "Time (days)", False
    value = field(dataset, "dts")
    if value is not None:
        value = np.asarray(value, dtype=float).reshape(-1)
        if len(value) == count - 1 and np.all(np.isfinite(value)) and np.all(value >= 0):
            return np.r_[0.0, np.cumsum(value)], "Time (days)", False
        if len(value) == count and np.all(np.isfinite(value)) and np.all(value >= 0):
            return np.cumsum(value), "Time (days)", False
    # T is a nominal generation horizon, not a substitute for random observation times.
    return np.arange(count), "Observation index", True


def load_path_data(run, analysis):
    bond = analysis.args.bond
    with np.load(run.path, allow_pickle=True) as npz:
        dataset = unpack(npz["dataset"])
        states = field(dataset, "states")
        true_eta = np.asarray(states[1], dtype=float)
        if not 0 <= bond < true_eta.shape[1]:
            raise ValueError(f"--bond={bond} out of range for D={true_eta.shape[1]} (zero-based).")
        params = mapping(npz["params"])
        coords, _ = iteration_coordinates(run.config, np.asarray(params["H"]).shape[0])
        if "references" in npz.files:
            reference = mapping(npz["references"])
            trajectories = reference["trajectory"]
        elif "trajectories" in npz.files:
            trajectories = npz["trajectories"]
        else:
            raise ValueError("No saved reference trajectories.")
        eta = np.asarray(trajectories[1])
        expected = len(coords) if run.config.get("saved_paths") is None else min(int(run.config["saved_paths"]), len(coords))
        if eta.ndim != 3 or eta.shape != (expected, *true_eta.shape):
            raise ValueError(f"Unexpected rolling trajectory shape {eta.shape}; expected {(expected, *true_eta.shape)}.")
        reference_iterations = coords[-len(eta):]
        eta = eta[reference_iterations >= int(run.config["burnin"]), :, bond]
        if len(eta) == 0:
            raise ValueError("No post-burn-in reference paths.")
        means = np.asarray(npz["standardisation_means"])
        scales = np.asarray(npz["standardisation_scales"])
        eta = means[bond] + scales[bond] * eta
        times, xlabel, fallback = observation_times(dataset, true_eta.shape[0])
        if fallback:
            analysis.note(f"{run.path.parent.name}: timestamps not stored in dataset; path plotted against observation index.")
        obs, indices, events = (np.asarray(v) for v in field(dataset, "data"))
        if len(obs) != len(times) or indices.shape != obs.shape or events.shape != obs.shape:
            raise ValueError("Observation arrays do not align with latent-state times.")
        truth_params = mapping(npz["true_params"])
        alpha = np.asarray(truth_params.get("alpha", 0.0))
        alpha = float(alpha) if alpha.ndim == 0 else float(alpha[bond])
        return times, xlabel, true_eta[:, bond], eta, obs, indices, events, alpha


def plot_paths(runs, analysis, directory, suffix):
    lookup = {r.meta["kernel"]: r for r in runs}
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True, layout="constrained")
    event_styles = {0: ("v", "#CC79A7", "D2C: Buy"),
                    1: ("^", "#E69F00", "D2C: Sell"),
                    2: ("s", "#777777", "D2D")}
    plotted = False
    xlabel = "Time"
    for ax, kernel in zip(axes, METHODS):
        ax.set_title(LABELS[kernel], loc="left", fontsize=10)
        if kernel not in lookup:
            ax.text(0.5, 0.5, "Result unavailable", ha="center", transform=ax.transAxes)
            continue
        try:
            times, xlabel, truth, draws, obs, indices, events, alpha = load_path_data(lookup[kernel], analysis)
        except (KeyError, ValueError, AttributeError) as error:
            analysis.note(f"Path panel skipped for {lookup[kernel].path.parent.name}: {error}")
            if analysis.args.strict:
                raise
            ax.text(0.5, 0.5, "Path data unavailable", ha="center", transform=ax.transAxes)
            continue
        lower = (1 - analysis.args.credible_mass) / 2
        lo, hi = np.quantile(draws, [lower, 1 - lower], axis=0)
        ax.fill_between(times, lo, hi, color=COLOURS[kernel], alpha=0.18,
                        label=f"{100 * analysis.args.credible_mass:g}% pointwise interval")
        ax.plot(times, truth, color="black", linestyle="--", linewidth=0.65, label="True η")
        ax.plot(times, draws.mean(axis=0), color=COLOURS[kernel], linewidth=0.9, label="Posterior mean η")
        for event in sorted(np.unique(events[indices == analysis.args.bond])):
            mask = (indices == analysis.args.bond) & (events == event)
            marker, colour, label = event_styles.get(int(event), ("o", "0.4", f"Observation type {event}"))
            if event == 2:
                ax.errorbar(times[mask], obs[mask], yerr=alpha, fmt=marker, markersize=2,
                            elinewidth=0.8, color=colour, alpha=0.9, label=label)
            else:
                ax.scatter(times[mask], obs[mask], marker=marker, s=16, color=colour, alpha=0.9, label=label)
        ax.set_ylabel(f"η, bond {analysis.args.bond + 1}")
        ax.grid(alpha=0.18)
        ax.text(0.99, 0.98, f"{len(draws)} retained paths", ha="right", va="top", transform=ax.transAxes, fontsize=8)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="lower left", fontsize=7, ncol=2)
        plotted = True
    axes[-1].set_xlabel(xlabel)
    if analysis.args.time_window:
        axes[-1].set_xlim(*analysis.args.time_window)
    fig.suptitle(f"Latent mid-yield reconstruction; {suffix.replace('_', ', ')}")
    if plotted:
        analysis.save(fig, directory, f"eta_paths_{suffix}")
    else:
        plt.close(fig)


def classified_pairs(pcs, tolerance):
    rows, cols = np.triu_indices(len(pcs), k=1)
    nonzero, zero = [], []
    for i, j in zip(rows, cols):
        pair = (int(i), int(j))
        (nonzero if abs(pcs[i, j]) > tolerance else zero).append(pair)
    return nonzero, zero


def sampled_zero_pairs(pcs, args):
    _, zero = classified_pairs(pcs, args.pc_zero_tol)
    if not zero or args.zero_trace_count == 0:
        return []
    rng = np.random.default_rng(args.trace_selection_seed)
    choices = rng.choice(len(zero), size=min(args.zero_trace_count, len(zero)), replace=False)
    return sorted(zero[int(i)] for i in choices)


def selected_pairs(pcs, args):
    """Compact truth-based selection; full nonzero collection is saved separately."""
    nonzero, _ = classified_pairs(pcs, args.pc_zero_tol)
    ordered = sorted(nonzero, key=lambda pair: (abs(pcs[pair]), pair))
    pairs = []
    if ordered:
        pairs = list(dict.fromkeys(ordered[i] for i in (0, len(ordered) // 2, len(ordered) - 1)))
    return pairs + sampled_zero_pairs(pcs, args)[:1]


def autocorrelation(values, max_lag):
    centred = np.asarray(values, dtype=float) - np.mean(values)
    denominator = np.dot(centred, centred)
    if denominator == 0:
        return np.full(max_lag + 1, np.nan)
    return np.array([np.dot(centred[:len(values) - lag], centred[lag:]) / denominator
                     for lag in range(max_lag + 1)])


def plot_selected_traces(runs, analysis, directory, suffix):
    pairs = selected_pairs(runs[0].pcs_truth, analysis.args)
    titles = ["H[1,1]"] + [f"Partial correlation [{i + 1},{j + 1}]" for i, j in pairs]
    truth = [runs[0].truth[0, 0]] + [runs[0].pcs_truth[i, j] for i, j in pairs]
    fig, axes = plt.subplots(len(titles), 3, figsize=(12, 2.5 * len(titles)),
                             squeeze=False, sharey="row", layout="constrained")
    acf_fig, acf_axes = plt.subplots(1, len(titles), figsize=(4.2 * len(titles), 3.6),
                                     squeeze=False, layout="constrained")
    lookup = {r.meta["kernel"]: r for r in runs}
    for col, kernel in enumerate(METHODS):
        axes[0, col].set_title(LABELS[kernel], fontsize=10)
        if kernel not in lookup:
            for ax in axes[:, col]:
                ax.set_axis_off()
            continue
        run = lookup[kernel]
        with np.load(run.path, allow_pickle=True) as npz:
            H = original_H(npz)
        coords, _ = iteration_coordinates(run.config, len(H))
        pcs = pc_history(H) if pairs else None
        series = [H[:, 0, 0]] + [pcs[:, i, j] for i, j in pairs]
        post = coords >= int(run.config["burnin"])
        for row, (values, label, target) in enumerate(zip(series, titles, truth)):
            ax = axes[row, col]
            ax.plot(coords, values, color=COLOURS[kernel], linewidth=0.9)
            ax.axvline(int(run.config["burnin"]), color="0.4", linestyle=":", linewidth=1)
            ax.axhline(target, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Gibbs iteration")
            ax.set_ylabel(label)
            ax.grid(alpha=0.18)
            kept = values[post]
            max_lag = min(analysis.args.acf_lags, max(0, len(kept) // 2))
            acf_axes[0, row].plot(np.arange(max_lag + 1) * int(run.config["thin"]),
                                  autocorrelation(kept, max_lag), color=COLOURS[kernel],
                                  marker="o", markersize=2.5, label=LABELS[kernel])
    for ax, title in zip(acf_axes[0], titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Lag (Gibbs iterations; retained samples)")
        ax.set_ylabel("Autocorrelation")
        ax.axhline(0, color="0.6", linewidth=0.6)
        ax.grid(alpha=0.18)
    acf_axes[0, 0].legend(fontsize=8)
    fig.suptitle("Selected traces: weak/medium/strong true dependencies and a true-zero pair")
    acf_fig.suptitle("Covariance/dependence mixing after burn-in")
    analysis.save(fig, directory, f"selected_traces_{suffix}")
    analysis.save(acf_fig, directory, f"selected_ACF_{suffix}")
    plot_pc_trace_collections(runs, analysis, directory, suffix)


def worst_mixing_pairs(runs, args):
    """Union of worst pairs per method; include unestimable/stuck entries first."""
    chosen = set()
    for run in runs:
        off = run.rows != run.cols
        pairs = list(zip(run.rows[off], run.cols[off]))
        values = run.pcs_ess[off]
        # Missing ESS because diagnostics were explicitly disabled is not a ranking.
        if args.no_ess or args.worst_pc_count == 0:
            continue
        order = np.argsort(np.where(np.isfinite(values), values, -np.inf), kind="stable")
        chosen.update((int(pairs[k][0]), int(pairs[k][1])) for k in order[:args.worst_pc_count])
    return sorted(chosen)


def plot_pc_trace_collections(runs, analysis, directory, suffix):
    """Paginate all nonzero PCs, sampled zeros, and worst-mixing pairs.

    Every page uses identical pairs and row-wise y limits for all three methods.
    Posterior PCs are formed from inverse(H) at each retained iteration. PDFs
    are combined into one multipage document per category; PNG/SVG are paged.
    """
    true_pcs = runs[0].pcs_truth
    nonzero, zero = classified_pairs(true_pcs, analysis.args.pc_zero_tol)
    if not zero:
        analysis.note(f"{suffix}: no true-zero PCs at tolerance {analysis.args.pc_zero_tol:g}; inverse(H) is dense even if H is sparse.")
    groups = [
        ("nonzero", nonzero, "All truly nonzero partial correlations"),
        ("zero", sampled_zero_pairs(true_pcs, analysis.args), "Reproducibly selected truly zero pairs"),
        ("worst_mixing", worst_mixing_pairs(runs, analysis.args), "Worst PC ESS: union across methods (diagnostic selection)"),
    ]
    # Invert once per method, then keep only off-diagonal entries.
    history = {}
    for run in runs:
        with np.load(run.path, allow_pickle=True) as npz:
            H = original_H(npz)
        coords, _ = iteration_coordinates(run.config, len(H))
        pcs = pc_history(H)
        history[run.meta["kernel"]] = (run, coords, pcs[:, run.rows, run.cols],
                                      {(int(i), int(j)): k for k, (i, j) in enumerate(zip(run.rows, run.cols))})
    selection_rows = []
    directory.mkdir(parents=True, exist_ok=True)
    for category, pairs, caption in groups:
        if not pairs:
            continue
        pdf_path = directory / f"{category}_pc_traces_{suffix}.pdf"
        pdf = PdfPages(pdf_path) if "pdf" in analysis.args.formats else None
        try:
            per_page = analysis.args.trace_pairs_per_page
            for page, start in enumerate(range(0, len(pairs), per_page), start=1):
                selected = pairs[start:start + per_page]
                fig, axes = plt.subplots(len(selected), 3, figsize=(12, 2.4 * len(selected)),
                                         squeeze=False, sharey="row", layout="constrained")
                for col, kernel in enumerate(METHODS):
                    axes[0, col].set_title(LABELS[kernel], fontsize=10)
                    if kernel not in history:
                        for ax in axes[:, col]:
                            ax.text(0.5, 0.5, "Result unavailable", ha="center", transform=ax.transAxes)
                        continue
                    run, coords, entries, index = history[kernel]
                    for row, pair in enumerate(selected):
                        i, j = pair
                        ax = axes[row, col]
                        ax.plot(coords, entries[:, index[pair]], color=COLOURS[kernel], linewidth=0.8)
                        ax.axvline(int(run.config["burnin"]), color="0.4", linestyle=":", linewidth=0.8)
                        ax.axhline(true_pcs[pair], color="black", linestyle="--", linewidth=0.8)
                        ax.set_ylabel(f"PC [{i + 1},{j + 1}]\ntrue={true_pcs[pair]:.3g}")
                        ax.set_xlabel("Gibbs iteration")
                        ax.grid(alpha=0.18)
                        value = run.pcs_ess[index[pair]]
                        if analysis.ess_enabled:
                            text = f"{analysis.args.ess_method} ESS={value:.1f}" if np.isfinite(value) else "ESS unavailable/stuck"
                            ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top", fontsize=7)
                total_pages = (len(pairs) + per_page - 1) // per_page
                fig.suptitle(f"{caption}\n{suffix.replace('_', ', ')}; page {page}/{total_pages}", fontsize=11)
                if pdf is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                analysis.save(fig, directory, f"{category}_pc_traces_{suffix}_page={page:03d}",
                              formats=[f for f in analysis.args.formats if f != "pdf"])
                for pair in selected:
                    row = {"category": category, "page": page, "i": pair[0], "j": pair[1],
                           "true_partial_correlation": float(true_pcs[pair]),
                           "true_nonzero": bool(abs(true_pcs[pair]) > analysis.args.pc_zero_tol),
                           "zero_tolerance": analysis.args.pc_zero_tol}
                    for kernel, (run, _, _, index) in history.items():
                        row[f"{kernel}_pc_ess"] = float(run.pcs_ess[index[pair]])
                    selection_rows.append(row)
        finally:
            if pdf is not None:
                pdf.close()
                analysis.outputs.append(str(pdf_path))
    write_csv(directory / f"pc_trace_selection_{suffix}.csv", selection_rows)


def grouped_rhat(runs, analysis):
    """Multi-chain check only for files with identical observed data and budget."""
    if not analysis.ess_enabled:
        return []
    groups = defaultdict(list)
    for run in runs:
        groups[(run.cohort, run.regime, run.meta["D"], run.meta["kernel"], run.dataset_id)].append(run)
    output = []
    for group in groups.values():
        first = group[0]
        row = {"configuration": cohort_name(first.cohort), "regime": first.regime,
               "D": first.meta["D"], "kernel": first.meta["kernel"],
               "dataset_id": first.dataset_id, "chains": len(group)}
        if len(group) == 1:
            row["max_rank_rhat"] = float("nan")
            output.append(row)
            continue
        seeds = [r.meta["seed"] for r in group]
        if len(set(seeds)) != len(seeds):
            analysis.note(f"{row}: repeated sampler seeds; multi-chain R-hat skipped.")
            row["max_rank_rhat"] = float("nan")
            output.append(row)
            continue
        chains = []
        for run in group:
            with np.load(run.path, allow_pickle=True) as npz:
                H = original_H(npz)
            coords, _ = iteration_coordinates(run.config, len(H))
            chains.append(H[coords >= int(run.config["burnin"])][:, first.rows, first.cols])
        draws = min(len(c) for c in chains)
        stacked = np.stack([c[:draws] for c in chains])
        if draws < 8:
            row["max_rank_rhat"] = float("nan")
        else:
            rhat = local_rank_rhat(stacked)
            row["max_rank_rhat"] = float(np.max(rhat)) if np.all(np.isfinite(rhat)) else float("nan")
            row["nonfinite_rhat_entries"] = int((~np.isfinite(rhat)).sum())
            if row["nonfinite_rhat_entries"] or row["max_rank_rhat"] > 1.01:
                analysis.note(f"{first.meta['kernel']}, D={first.meta['D']}, {first.regime}: chains do not establish convergence (max R-hat={row['max_rank_rhat']:.3g}).")
        output.append(row)
    return output


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def parser():
    out = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    out.add_argument("--root", type=Path, default=Path.cwd(), help="Default results=ROOT/results; output=ROOT/plots.")
    out.add_argument("--results-dir", action="append", default=[], metavar="[LABEL=]PATH",
                     help="Repeat to compare observation-frequency regimes. Paths are used as given.")
    out.add_argument("--plot-dir", type=Path, default=None)
    out.add_argument("--dims", type=int, nargs="+", default=None)
    out.add_argument("--seeds", type=int, nargs="+", default=None)
    out.add_argument("--N", type=int, default=None)
    out.add_argument("--samples", type=int, default=None)
    out.add_argument("--burnin", type=int, default=None)
    out.add_argument("--plot-D", type=int, default=20)
    out.add_argument("--plot-seed", type=int, default=1234)
    out.add_argument("--bond", type=int, default=0, help="Zero-based bond index for the illustrative path.")
    out.add_argument("--time-window", type=float, nargs=2, metavar=("START", "END"), default=None)
    out.add_argument("--credible-mass", type=float, default=0.9)
    out.add_argument("--acf-lags", type=int, default=30, help="Maximum lag measured in retained draws.")
    out.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"])
    out.add_argument("--dpi", type=int, default=200)
    out.add_argument("--no-ess", action="store_true")
    out.add_argument("--ess-method", choices=("bulk", "mean"), default="bulk",
                     help="Local split rank-normalised ESS (bulk), or raw split-chain ESS (mean).")
    out.add_argument("--pc-zero-tol", type=float, default=1e-8,
                     help="Absolute true-PC tolerance for classifying zero/nonzero dependencies.")
    out.add_argument("--trace-pairs-per-page", type=int, default=4,
                     help="Rows per page in the full PC trace collections; every true nonzero is included.")
    out.add_argument("--zero-trace-count", type=int, default=4,
                     help="Number of reproducibly selected true-zero PC traces.")
    out.add_argument("--worst-pc-count", type=int, default=3,
                     help="Worst PC ESS entries per method; their union is plotted for all methods.")
    out.add_argument("--trace-selection-seed", type=int, default=1234)
    out.add_argument("--strict", action="store_true", help="Fail on malformed saved files rather than reporting and skipping.")
    return out


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < args.credible_mass < 1 or args.bond < 0 or args.acf_lags < 0:
        raise SystemExit("Require 0<credible-mass<1, bond>=0, and acf-lags>=0.")
    if (not np.isfinite(args.pc_zero_tol) or args.pc_zero_tol < 0 or
            args.trace_pairs_per_page < 1 or args.zero_trace_count < 0 or args.worst_pc_count < 0):
        raise SystemExit("Require finite pc-zero-tol>=0, trace-pairs-per-page>=1, and trace counts>=0.")
    if args.time_window and args.time_window[0] >= args.time_window[1]:
        raise SystemExit("--time-window START must be less than END.")
    args.root = args.root.expanduser().resolve()
    args.plot_dir = (args.plot_dir or args.root / "plots").expanduser().resolve()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    analysis = Analysis(args)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    roots = []
    for spec in args.results_dir or [str(args.root / "results")]:
        if "=" in spec and not Path(spec).expanduser().exists():
            label, path = spec.split("=", 1)
            roots.append((label, Path(path).expanduser().resolve()))
        else:
            roots.append((None, Path(spec).expanduser().resolve()))
    runs, skipped, seen = [], [], set()
    for label, directory in roots:
        if not directory.is_dir():
            raise SystemExit(f"Results directory not found: {directory}")
        for path in sorted(directory.rglob("data.npz")):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            meta = name_metadata(path)
            if meta.get("kernel") not in METHODS or any(k not in meta for k in ("D", "T", "steps", "N", "seed")):
                skipped.append({"path": str(path), "reason": "Not a recognised experiment directory"})
                continue
            if meta["backward"] != MODES[meta["kernel"]]:
                skipped.append({"path": str(path), "reason": "Backward mode differs from the agreed comparison"})
                continue
            if args.dims and meta["D"] not in args.dims or args.seeds and meta["seed"] not in args.seeds:
                continue
            if any(getattr(args, k) is not None and meta.get(k) != getattr(args, k) for k in ("N", "samples", "burnin")):
                continue
            print(f"Loading {path.parent.name}")
            try:
                runs.append(analyse_run(path, label, meta, analysis))
            except Exception as error:
                if args.strict:
                    raise
                reason = f"{type(error).__name__}: {error}"
                skipped.append({"path": str(path), "reason": reason})
                analysis.note(f"Skipped {path.parent.name}: {reason}")
    if not runs:
        write_csv(args.plot_dir / "skipped_results.csv", skipped)
        raise SystemExit("No compatible results loaded. See skipped_results.csv; run from the original repo/environment for pickled datasets.")
    assign_regimes(runs)
    analysis.note("No runtime is saved: ESS/second cannot be calculated from these files.")
    analysis.note("Replacement rates use the saved rolling aux['replaced'] diagnostic; active-coordinate semantics cannot be verified from these attachments.")
    analysis.note("Current filenames do not encode phi/full-inference, and config does not save them; keep different model/prior settings in separate result roots.")
    grouped = defaultdict(list)
    for run in runs:
        grouped[run.cohort].append(run)
    for cohort, members in grouped.items():
        directory = args.plot_dir / cohort_name(cohort)
        plot_scaling(members, analysis, directory)
        for regime, representative in representative_sets(members, analysis):
            representative_directory = directory / slug(regime)
            first = representative[0]
            suffix = f"D={first.meta['D']}_seed={first.meta['seed']}"
            plot_heatmaps(representative, analysis, representative_directory, suffix)
            plot_paths(representative, analysis, representative_directory, suffix)
            plot_selected_traces(representative, analysis, representative_directory, suffix)
    for cohort, regime, D in {(r.cohort, r.regime, r.meta["D"]) for r in runs}:
        subset = [r for r in runs if (r.cohort, r.regime, r.meta["D"]) == (cohort, regime, D)]
        missing = set(METHODS) - {r.meta["kernel"] for r in subset}
        if missing:
            analysis.note(f"{regime}, D={D}: missing comparison methods {', '.join(sorted(missing))}.")
        seeds = {r.meta["seed"] for r in subset}
        datasets = {r.dataset_id for r in subset}
        if len(seeds) > 1 and len(datasets) == 1:
            analysis.note(f"{regime}, D={D}: {len(seeds)} sampler seeds share one synthetic dataset; bands show sampler variability.")
    chain_rows = grouped_rhat(runs, analysis)
    summary = [{"path": str(r.path), "configuration": cohort_name(r.cohort),
                "regime": r.regime, **r.meta, **r.metrics} for r in runs]
    write_csv(args.plot_dir / "metrics.csv", summary)
    write_csv(args.plot_dir / "entrywise_ESS.csv", (
        {"path": str(r.path), "regime": r.regime, "kernel": r.meta["kernel"],
         "D": r.meta["D"], "seed": r.meta["seed"], "i": int(i), "j": int(j),
         "ess": float(ess), "ess_method": args.ess_method,
         "bulk_ess": float(ess) if args.ess_method == "bulk" else float("nan"),
         "relative_ess": float(ess / r.metrics["retained_posterior_draws"]),
         "thin": r.config["thin"]}
        for r in runs for i, j, ess in zip(r.rows, r.cols, r.ess)))
    write_csv(args.plot_dir / "partial_correlation_ESS.csv", (
        {"path": str(r.path), "regime": r.regime, "kernel": r.meta["kernel"],
         "D": r.meta["D"], "seed": r.meta["seed"], "i": int(i), "j": int(j),
         "true_pc": float(r.pcs_truth[i, j]),
         "true_nonzero": bool(abs(r.pcs_truth[i, j]) > args.pc_zero_tol),
         "ess": float(ess), "ess_method": args.ess_method,
         "relative_ess": float(ess / r.metrics["retained_posterior_draws"]),
         "thin": r.config["thin"]}
        for r in runs for i, j, ess in zip(r.rows, r.cols, r.pcs_ess) if i != j))
    write_csv(args.plot_dir / "chain_diagnostics.csv", chain_rows)
    write_csv(args.plot_dir / "skipped_results.csv", skipped)
    manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "results": [str(r.path) for r in runs], "figures": analysis.outputs,
                "notes": analysis.notes, "skipped": skipped}
    (args.plot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.plot_dir / "analysis_notes.txt").write_text(__doc__ + "\n\nRun notes:\n" + "\n".join(analysis.notes) + "\n")
    print(f"Saved {len(analysis.outputs)} figure files from {len(runs)} runs to {args.plot_dir}")


if __name__ == "__main__":
    main()
