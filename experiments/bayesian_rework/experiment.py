import argparse

import jax.numpy as jnp
import jax.random as jr

from jax.random import PRNGKey

from rbsmc.utils.common import force_move
from rbsmc.utils.resamplings import killing
from rbsmc.bayesian.smc import SMC
from rbsmc.bayesian.training_rework import BayesianInference, Config

from bayesian_rework.data import get_data, get_prior_params
from bayesian_rework.kernels import CSMC

parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=100)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)

parser.add_argument("--burnin", type=int, default=500)
parser.add_argument("--samples", dest="samples", type=int, default=500)

parser.add_argument("--phi", type=float, default=0.8)
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
csmc = CSMC(N=args.N, D=args.D, dts=DTs)
kwargs = dict(resampling_func=killing, backward=args.backward, ancestor_move_func=force_move) 
KERNEL = SMC(
    fk=csmc, 
    conditional=args.conditional,
    kwargs=kwargs
)

# GIBBS CONFIG
GIBBS = None # TODO

# INFERENCE CONFIG
CONFIG = Config(samples=args.samples, burnin=args.burnin, seed=args.seed)
SAMPLER = BayesianInference(smc=KERNEL, gibbs=GIBBS, config=CONFIG)


def one_experiment(key: PRNGKey):

    # generate data
    key, data_key = jr.split(key)
    true_xs, data = get_data(key=data_key, dim=args.D, dts=DTs, **PRIOR_PARAMS)

    # run particle Gibbs
    samples, ancestors, params, replacement_rates = SAMPLER.run(data)

    return samples, ancestors, params, replacement_rates, SAMPLER.energies


