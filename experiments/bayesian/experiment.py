import argparse
import os
from dataclasses import asdict

import numpy as np

import jax.random as jr

import jax
jax.config.update('jax_enable_x64', True)

from jax.random import PRNGKey

from rbsmc.utils.common import force_move
from rbsmc.utils.resamplings import killing
from rbsmc.bayesian.smc import SMC
from rbsmc.bayesian.training import ParticleGibbs, Config
from rbsmc.bayesian.gibbs import Gibbs

from experiments.bayesian.data import get_data, get_model_params
from experiments.bayesian.kernels import KernelType
from experiments.bayesian.gibbs import make_blocks
from experiments.bayesian.dataset import estimate_params_from_data


parser = argparse.ArgumentParser()

parser.add_argument("--M", dest="M", type=int, default=1)  # number of chains
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1
parser.add_argument("--T", dest="T", type=int, default=500)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--steps", type=int, default=499)
parser.add_argument("--kernel", type=int, default=1)
parser.add_argument("--burnin", type=int, default=500)
parser.add_argument("--samples", dest="samples", type=int, default=500)
parser.add_argument("--thin", type=int, default=1)
parser.add_argument("--saved-paths", type=int, default=None)
parser.add_argument("--phi", type=float, default=0.1)
parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--full-inference", action='store_true')
parser.add_argument('--no-full-inference', dest='full_inference', action='store_false')
parser.set_defaults(full_inference=False)

parser.add_argument("--conditional", action="store_true")
parser.add_argument("--unconditional", dest="conditional", action="store_false")
parser.set_defaults(conditional=True)

parser.add_argument("--backward-mode", choices=("ancestral", "ffbsi", "reduced"), default=None)
parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--debug", action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

args = parser.parse_args()

# RNG
KEY = PRNGKey(0)  # same every time
INIT_KEY, EXPERIMENT_KEY = jr.split(KEY)

# INIT TRUE PARAMETERS
MODEL_PARAMS, DTs = get_model_params(INIT_KEY,
                                     args.D,
                                     args.T,
                                     args.steps,
                                     args.phi)


# SMC CONFIG
kernel = KernelType(args.kernel).kernel_maker(N=args.N, D=args.D, dts=DTs)
BACKWARD_MODE = args.backward_mode or ("ffbsi" if args.backward else "ancestral")
if BACKWARD_MODE == "reduced" and kernel.name != "RB_CSMC":
    parser.error("--backward-mode=reduced is available only for the RB_CSMC kernel.")

kwargs = dict(resampling_func=killing, ancestor_move_func=force_move)
if kernel.name == "RB_CSMC":
    kwargs["backward_mode"] = BACKWARD_MODE
else:
    kwargs["backward"] = BACKWARD_MODE == "ffbsi"

KERNEL = SMC(
    fk=kernel,
    conditional=args.conditional,
    kwargs=kwargs
)

# GIBBS CONFIG
BLOCKS = make_blocks(D=args.D, full_inference=args.full_inference)
GIBBS = Gibbs(blocks=BLOCKS)

# INFERENCE CONFIG
CONFIG = Config(
    samples=args.samples,
    burnin=args.burnin,
    seed=args.seed,
    thin=args.thin,
    saved_paths=args.saved_paths,
)
SAMPLER = ParticleGibbs(smc=KERNEL, gibbs=GIBBS, config=CONFIG)

print(f"""
========================
Configuration
    - D:                 {args.D}
    - T:                 {args.T}
    - steps:             {args.steps}
    - kernel:            {kernel.name}
    - backward mode:     {BACKWARD_MODE}
    - full inference:    {args.full_inference}
    - thin:              {CONFIG.thin}
    - saved paths:       {CONFIG.saved_paths}
========================
""")


def one_experiment(key: PRNGKey):

    # generate data
    key, data_key = jr.split(key)
    dataset = get_data(key=data_key, dim=args.D, dts=DTs, params=MODEL_PARAMS)
    estimated_params = {}
    if not args.full_inference:
        estimated_params = estimate_params_from_data(dataset=dataset)

    dataset.params = {**dataset.params, **estimated_params}
    scaled_dataset = dataset.standardised_data

    # run particle Gibbs. Passing prior params uses true params only for those without Gibbs blocks
    references, params, replacement_rates = SAMPLER.run(
        scaled_dataset.data, DTs, scaled_dataset.params
    )
    return references, params, replacement_rates, SAMPLER.energies, dataset, scaled_dataset, estimated_params


def _pack_object(value):
    """Store a structured Python value as one NPZ object without coercing it."""
    packed = np.empty((), dtype=object)
    packed[()] = value
    return packed


def _serialise_reference(reference_history):
    """Convert a concrete reference NamedTuple to a stable, plain mapping."""
    return {"type": type(reference_history).__name__, **reference_history._asdict()}


if __name__ == "__main__":

    references, params, replacement_rates, energies, dataset, scaled_dataset, estimated_params = one_experiment(EXPERIMENT_KEY)

    # save results
    if not os.path.exists("results"):
        os.mkdir("results")

    experiment_name = "kernel={},D={},T={},steps={},phi={},N={},samples={},burnin={},full-inference={},conditional={},seed={},backward-mode={}"
    experiment_name = experiment_name.format(
        kernel.name,
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
        BACKWARD_MODE,
    )

    dirpath = f"results/{experiment_name}"
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    datapath = f"{dirpath}/data.npz"
    np.savez_compressed(
        datapath,
        references=_pack_object(_serialise_reference(references)),
        params=params,
        energies=energies,
        replacement_rates=replacement_rates,
        dataset=dataset,
        true_params=MODEL_PARAMS,
        estimated_params=estimated_params,
        standardisation_means=scaled_dataset.means,
        standardisation_scales=scaled_dataset.stds,
        config=_pack_object(asdict(CONFIG)),
    )
