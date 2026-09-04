import argparse
import os

import numpy as np

import jax.numpy as jnp
import jax.random as jr

import jax
jax.config.update('jax_enable_x64', True)

from jax.random import PRNGKey

from rbsmc.utils.common import force_move
from rbsmc.utils.resamplings import killing
from rbsmc.bayesian.smc import SMC
from rbsmc.bayesian.training import ParticleGibbs, Config
from rbsmc.bayesian.gibbs import Gibbs

from experiments.bayesian.data import get_data, get_prior_params
from experiments.bayesian.kernels import KernelType
from experiments.bayesian.gibbs import make_blocks
from experiments.bayesian.utils import print_z_diagnostics


parser = argparse.ArgumentParser()

parser.add_argument("--M", dest="M", type=int, default=1)  # number of chains

parser.add_argument("--T", dest="T", type=int, default=500)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--steps", type=int, default=499)

parser.add_argument("--kernel", type=int, default=1)

parser.add_argument("--burnin", type=int, default=500)
parser.add_argument("--samples", dest="samples", type=int, default=500)

parser.add_argument("--phi", type=float, default=0.1)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)

parser.add_argument("--seed", dest="seed", type=int, default=1234)

parser.add_argument("--conditional", action="store_true")
parser.add_argument("--unconditional", dest="conditional", action="store_false")
parser.set_defaults(conditional=True)

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

parser.add_argument("--debug", action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

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


# RNG
KEY = PRNGKey(0)  # same every time
INIT_KEY, EXPERIMENT_KEY = jr.split(KEY)

# INIT TRUE PARAMETERS
PRIOR_PARAMS, DTs = get_prior_params(INIT_KEY, 
                                     args.D, 
                                     args.T, 
                                     args.steps, 
                                     args.phi, 
                                     args.log_var)


# SMC CONFIG
kernel = KernelType(args.kernel).kernel_maker(N=args.N, D=args.D, dts=DTs)
kwargs = dict(resampling_func=killing, backward=args.backward, ancestor_move_func=force_move) 
KERNEL = SMC(
    fk=kernel, 
    conditional=args.conditional,
    kwargs=kwargs
)

# GIBBS CONFIG
BLOCKS = make_blocks(D=args.D, infer_H=args.infer_H, infer_m0=args.infer_m0, infer_H0=args.infer_H0)
GIBBS = Gibbs(blocks=BLOCKS)

# INFERENCE CONFIG
CONFIG = Config(samples=args.samples, burnin=args.burnin, seed=args.seed)
SAMPLER = ParticleGibbs(smc=KERNEL, gibbs=GIBBS, config=CONFIG)

print(f"""
========================
Configuration
    - D:         {args.D}
    - T:         {args.T}
    - steps:     {args.steps}
    - kernel:    {kernel.name}
    - infer H:   {args.infer_H}
    - infer m0:  {args.infer_m0}
    - infer H0:  {args.infer_H0}
========================
""")

def one_experiment(key: PRNGKey):

    # generate data
    key, data_key = jr.split(key)
    dataset = get_data(key=data_key, dim=args.D, dts=DTs, params=PRIOR_PARAMS)
    # print_z_diagnostics(dataset)

    scaled_dataset = dataset.standardised_data
    scaled_params = dataset.standardised_params

    # run particle Gibbs. Passing prior params uses true params only for those without Gibbs blocks
    samples, ancestors, params, replacement_rates = SAMPLER.run(scaled_dataset, DTs, scaled_params)
    return samples, ancestors, params, replacement_rates, SAMPLER.energies, dataset


if __name__ == "__main__": 

    samples, As, params, replacement_rates, energies, dataset = one_experiment(EXPERIMENT_KEY)

    # save results
    if not os.path.exists("results"):
        os.mkdir("results")

    experiment_name = "kernel={},D={},T={},steps={},phi={},log-var={},N={},s={},b={},inf-H={},inf-m0={},inf-H0={},cond={},seed={}"
    experiment_name = experiment_name.format(
        kernel.name,
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
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    datapath = f"{dirpath}/data.npz"
    np.savez_compressed(
        datapath,
        trajectories=samples,
        ancestors=As,
        params=params,
        energies=energies,
        replacement_rates=replacement_rates,
        dataset=dataset,
        true_params=PRIOR_PARAMS,
        standardisation_means=dataset.means,
        standardisation_scales=dataset.stds,
    )
