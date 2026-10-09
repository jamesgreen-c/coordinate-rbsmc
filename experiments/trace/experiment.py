import argparse
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)

from jax import tree_util

from dataclasses import asdict, replace
from pathlib import Path

from rbsmc.utils.common import force_move
from rbsmc.utils.resamplings import killing
from rbsmc.smc import SMC
from rbsmc.bayesian.training import ParticleGibbs, Config
from rbsmc.bayesian.gibbs import Gibbs

from experiments.trace.data import get_data
from experiments.kernels import KernelType
from experiments.gibbs import make_blocks


parser = argparse.ArgumentParser()

parser.add_argument("--M", dest="M", type=int, default=1)   # number of chains
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

parser.add_argument("--kernel", type=int, default=1)
parser.add_argument("--backward-mode", choices=("ancestral", "ffbsi", "reduced"), default=None)
parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--burnin", type=int, default=500)
parser.add_argument("--samples", dest="samples", type=int, default=500)
parser.add_argument("--thin", type=int, default=1)
parser.add_argument("--saved-paths", type=int, default=None)

parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--debug", action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument("--root", type=Path, default=Path.cwd())

args = parser.parse_args()


# RESULTS DIR
RESULTS_ROOT = args.root.expanduser().resolve() / "results"

# DATASET
DATASET = get_data("files/cleaned_trace.csv")
SCALED_DATASET = DATASET.standardised_data

# SMC CONFIG
kernel = KernelType(args.kernel).kernel_maker(N=args.N, D=DATASET.D, dts=DATASET.dts)

BACKWARD_MODE = args.backward_mode or ("ffbsi" if args.backward else "ancestral")
if BACKWARD_MODE == "reduced" and kernel.name != "RB_CSMC":
    parser.error("--backward-mode=reduced is available only for the RB_CSMC kernel.")

kwargs = dict(resampling_func=killing, ancestor_move_func=force_move)
if kernel.name == "RB_CSMC":
    kwargs["backward_mode"] = BACKWARD_MODE
else:
    kwargs["backward"] = BACKWARD_MODE == "ffbsi"

KERNEL = SMC(fk=kernel, conditional=True, kwargs=kwargs)

# INFERENCE CONFIG
CONFIG = Config(
    samples=args.samples,
    burnin=args.burnin,
    seed=args.seed,
    thin=args.thin,
    saved_paths=args.saved_paths,
)

print(f"""
========================
Configuration
    - total trades:      {DATASET.data[0].shape[0]}
    - total bonds:       {DATASET.D}
    - kernel:            {kernel.name}
    - backward mode:     {BACKWARD_MODE}
    - thin:              {CONFIG.thin}
    - saved paths:       {CONFIG.saved_paths if CONFIG.saved_paths is not None else "all"}
========================
""")


def experiment():

    # gibbs config
    BLOCKS = make_blocks(D=DATASET.D, full_inference=True)
    GIBBS = Gibbs(blocks=BLOCKS)

    references = []
    params = []
    replacement_rates = []
    run_times = []
    sampling_times = []

    for m in range(args.M):

        # distinct sampler seed for each chain
        config_m = replace(CONFIG, seed=CONFIG.seed + m)
        SAMPLER = ParticleGibbs(smc=KERNEL, gibbs=GIBBS, config=config_m)

        references_m, params_m, replacement_rates_m, run_times_m = SAMPLER.run(
            SCALED_DATASET.data, 
            SCALED_DATASET.dts, 
            {}
        )
        
        references.append(tree_util.tree_map(np.asarray, references_m))
        params.append(tree_util.tree_map(np.asarray, params_m))
        replacement_rates.append(np.asarray(replacement_rates_m))
        run_times.append(run_times_m[-1])
        sampling_times.append(run_times_m[-1] - run_times_m[args.burnin])

    # every array has a leading chain dimension, including when M=1
    references = tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *references)
    params = tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *params)
    replacement_rates = np.stack(replacement_rates, axis=0)
    return references, params, replacement_rates, run_times, sampling_times


def _pack_object(value):
    """Store a structured Python value as one NPZ object without coercing it."""
    packed = np.empty((), dtype=object)
    packed[()] = value
    return packed


def _serialise_reference(reference_history):
    """Convert a concrete reference NamedTuple to a stable, plain mapping."""
    return {"type": type(reference_history).__name__, **reference_history._asdict()}


if __name__ == "__main__":

    references, params, replacement_rates, run_times, sampling_times = experiment()

    # save results
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    experiment_name = "kernel={},N={},samples={},burnin={},seed={},backward={}"
    experiment_name = experiment_name.format(
        kernel.name,
        args.N,
        args.samples,
        args.burnin,
        args.seed,
        BACKWARD_MODE,
    )

    DIRPATH = RESULTS_ROOT / experiment_name
    DIRPATH.mkdir(parents=True, exist_ok=True)
    DATAPATH = DIRPATH / "data.npz"

    np.savez_compressed(
        DATAPATH,
        references=_pack_object(_serialise_reference(references)),
        params=params,
        replacement_rates=replacement_rates,
        dataset=DATASET,
        standardisation_means=SCALED_DATASET.means,
        standardisation_scales=SCALED_DATASET.stds,
        config=_pack_object(asdict(CONFIG)),
        run_seconds=run_times,
        sampling_seconds=sampling_times
    )
