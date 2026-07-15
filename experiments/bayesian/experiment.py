import argparse
import os

import numpy as np

import jax
import jax.random as jr
from jax.tree_util import tree_map

from rbsmc.utils.common import force_move, barker_move
from rbsmc.utils.resamplings import killing, multinomial

from rbsmc.expmax.free_energy import constructor as free_energy_constructor
from rbsmc.bayesian.training import BayesianInference, Config

from experiments.bayesian import prior
from experiments.bayesian.kernels import KernelType

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=100)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)

parser.add_argument("--burnin", type=int, default=500)
parser.add_argument("--samples", dest="samples", type=int, default=500)

parser.add_argument("--phi", type=float, default=0.8)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)

parser.add_argument("--kernel", dest="kernel", type=int, default=KernelType.CSMC)
parser.add_argument("--style", dest="style", default="bootstrap")

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

# CONFIG
CONFIG = Config(burnin=args.burnin, samples=args.samples, seed=args.seed, debug=args.debug)
kernel_type = KernelType(args.kernel)
param_key, experiment_key = jr.split(jr.PRNGKey(args.seed))

# TRUE PRIOR PARAMETERS
(A, CHOL_Q0, CHOL_H0, CHOL_Q, CHOL_H, 
 BETA, DELTA, LLAMBDA, TAU, CHOL_R, 
 PSI, ALPHA, DTs) = prior.get_prior_params(param_key, args.D, args.T, args.steps, args.phi, args.log_var)


def one_experiment(key):

    # sample true data
    _get_data = lambda _k: prior.get_data(
        _k, args.D, DTs, 
        A, PSI, CHOL_Q0, CHOL_Q, CHOL_H0, CHOL_H, CHOL_R, ALPHA,
        sparsity_factor=1.0
    )
    true_xs, data, *_ = _get_data(key)
    true_xs = true_xs                     # xs needs leading dimension for free energy
    data = data[None, ...]                # data needs leading dimension for free energy

    # setup smc kernel
    kernel, kernel_init = kernel_type.kernel_maker(
        N=args.N,
        dts=DTs,
        conditional=args.conditional,
        resampling_func=killing,
        backward=args.backward,
        ancestor_move_func=force_move,
        style=args.style,
        sweeps=1
    )

    # construct loss function
    posterior_fn, loss_fn = free_energy_constructor(
        prior=prior,
        smc_init=kernel_init,
        smc=kernel,
        n_samples=1,
        dts=DTs
    )

    # define prior init - only learn A
    prior_init = lambda _: prior.init(
        dim=args.D,
        A=A,
        tau=TAU,
        llambda=LLAMBDA,
        chol_Q0=CHOL_Q0,
        chol_H0=CHOL_H0,
        chol_Q=CHOL_Q,
        chol_R=CHOL_R,
        psi=PSI,
        alpha=ALPHA,
        dts=DTs,
    )

    # define and run Bayesian inference
    trainer = BayesianInference(
        posterior_function=posterior_fn,
        loss_function=loss_fn,
        prior_init=prior_init,
        prior=prior,
        config=CONFIG,
    )
    trainer.fit(data)

    return trainer.loss_hist, trainer.param_hist, trainer.sample_hist, trainer.replacement_rates, true_xs, data

loss_history, param_hist, sample_hist, replacement_rates, true_xs, data = one_experiment(experiment_key)

# save results
if not os.path.exists("results"):
    os.mkdir("results")

experiment_name = "kernel={},D={},T={},phi={},log-var={},N={},samples={},burnin={},conditional={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name,
    args.D,
    args.T,
    args.phi,
    args.log_var,
    args.N,
    args.samples,
    args.burnin,
    args.conditional,
    args.seed,
)

dirpath = f"results/{experiment_name}"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
np.savez_compressed(
    datapath,
    param_hist=param_hist,
    sample_hist=sample_hist,
    true_A=A,
    true_beta=BETA,
    true_delta=DELTA,
    true_tau=TAU,
    loss_history=loss_history,
    replacement_rates=replacement_rates,
    xs=true_xs,
    data=data,
)
