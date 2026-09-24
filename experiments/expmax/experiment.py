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

from rbsmc.smc import SMC
from rbsmc.expmax.training import MonteCarloEM, Config
from rbsmc.expmax.free_energy import FreeEnergy

from experiments.expmax.data import get_data, get_model_params
from experiments.expmax.kernels import KernelType
from experiments.expmax.dataset import estimate_params_from_data
from experiments.expmax.prior import CorporateBondPrior


parser = argparse.ArgumentParser()

parser.add_argument("--M", dest="M", type=int, default=1)  # number of chains
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1
parser.add_argument("--T", dest="T", type=int, default=500)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--steps", type=int, default=499)
parser.add_argument("--kernel", type=int, default=1)
parser.add_argument("--num-iter", dest="num_iter", type=int, default=1000)
parser.add_argument("--num-samples", dest="num_samples", type=int, default=1)
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

# FREE ENERGY
KERNEL = SMC(
    fk=kernel,
    conditional=args.conditional,
    kwargs=kwargs
)
PRIOR = CorporateBondPrior(D=args.D)
MODEL = FreeEnergy(prior=PRIOR, smc=KERNEL)

# INFERENCE CONFIG
CONFIG = Config(
    num_iter=args.num_iter,
    num_samples=args.num_samples,
    seed=args.seed,
    thin=args.thin,
    saved_paths=args.saved_paths,
)
SAMPLER = MonteCarloEM(config=CONFIG, model=MODEL)

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
    # estimated_params = {}
    # if not args.full_inference:
    #     estimated_params = estimate_params_from_data(dataset=dataset)

    # dataset.params = {**dataset.params, **estimated_params}
    scaled_dataset = dataset.standardised_data

    # run particle Gibbs. Passing prior params uses true params only for those without Gibbs blocks
    references, params, replacement_rates = SAMPLER.run(
        scaled_dataset.data, DTs, scaled_dataset.params
    )
    return references, params, replacement_rates, SAMPLER.energies, dataset, scaled_dataset# , estimated_params


def _pack_object(value):
    """Store a structured Python value as one NPZ object without coercing it."""
    packed = np.empty((), dtype=object)
    packed[()] = value
    return packed


def _serialise_reference(reference_history):
    """Convert a concrete reference NamedTuple to a stable, plain mapping."""
    return {"type": type(reference_history).__name__, **reference_history._asdict()}


if __name__ == "__main__":

    references, params, replacement_rates, energies, dataset, scaled_dataset = one_experiment(EXPERIMENT_KEY)

    # save results
    if not os.path.exists("results"):
        os.mkdir("results")

    experiment_name = "kernel={},D={},T={},steps={},phi={},N={},num-iter={},full-inference={},conditional={},seed={},backward-mode={}"
    experiment_name = experiment_name.format(
        kernel.name,
        args.D,
        args.T,
        args.steps,
        args.phi,
        args.N,
        args.num_iter,
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
        # estimated_params=estimated_params,
        standardisation_means=scaled_dataset.means,
        standardisation_scales=scaled_dataset.stds,
        config=_pack_object(asdict(CONFIG)),
    )
