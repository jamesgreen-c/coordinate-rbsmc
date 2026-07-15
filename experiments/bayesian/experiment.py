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
parser.add_argument("--burnin", type=int, default=1)

parser.add_argument("--phi", type=float, default=0.8)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)

parser.add_argument("--independent", action="store_true")
parser.set_defaults(independent=False)

parser.add_argument("--n-series", dest="n_series", type=int, default=2048)
parser.add_argument("--split", dest="split", type=float, default=0.8)

parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
parser.add_argument("--num-iter", dest="num_iter", type=int, default=10_000)
parser.add_argument("--n-samples", dest="n_samples", type=int, default=64)
parser.add_argument("--n-sweeps", dest="sweeps", type=int, default=2)

parser.add_argument("--prior-lr", dest="prior_lr", type=float, default=5e-5)

parser.add_argument("--target", dest="target", type=int, default=75)
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

# TRAINING CONFIG
CONFIG = Config(batch_size=args.batch_size, num_iter=args.num_iter)
SPLIT =  int(args.n_series // (100 / (args.split * 100)))
STATIONARY = True

# PRIOR PARAMETERS
kernel_type = KernelType(args.kernel)
param_key, experiment_key = jr.split(jr.PRNGKey(args.seed))

(A, CHOL_Q0, CHOL_H0, CHOL_Q, CHOL_H, 
 BETA, DELTA, LLAMBDA, TAU, CHOL_R, 
 PSI, ALPHA, DTs) = prior.get_prior_params(param_key, args.D, args.T, args.steps, args.phi, args.log_var)


def one_experiment(key):
    data_key, test_key = jr.split(key)

    # sample true data
    _get_data = lambda _k: prior.get_data(
        _k, args.D, DTs, 
        A, PSI, CHOL_Q0, CHOL_Q, CHOL_H0, CHOL_H, CHOL_R, ALPHA,
        sparsity_factor=1.0
    )
    data_keys = jr.split(data_key, args.n_series)
    true_xs, data, *_ = jax.vmap(_get_data)(data_keys)

    train_xs, train_data = tree_map(lambda x: x[:SPLIT], true_xs), tree_map(lambda d: d[:SPLIT], data)
    test_xs, test_data = tree_map(lambda x: x[SPLIT:], true_xs), tree_map(lambda d: d[SPLIT:], data)

    # setup smc kernel
    kernel, kernel_init = kernel_type.kernel_maker(
        N=args.N,
        dts=DTs,
        conditional=args.conditional,
        resampling_func=killing,
        backward=args.backward,
        ancestor_move_func=force_move,
        style=args.style,
        sweeps=args.sweeps
    )

    # construct loss function
    posterior_fn, loss_fn = free_energy_constructor(
        prior=prior,
        smc_init=kernel_init,
        smc=kernel,
        n_samples=args.n_samples,
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
        stationary=STATIONARY
    )

    # define and fit trainer
    trainer = BayesianInference(
        posterior_function=posterior_fn,
        loss_function=loss_fn,
        prior_init=prior_init,
        prior=prior,
        config=CONFIG,
        # stabilise_function=prior.stabilise_params
    )
    trainer.fit(train_data)

    # test
    test_samples = trainer.apply(test_key, tree_map(lambda d: d[:32], test_data), inits=None, num_iter=10)
    return trainer.loss_hist, trainer.best_params, trainer.params, trainer.replacement_rates, test_xs, test_samples, test_data

loss_history, best_params, final_params, replacement_rates, test_xs, test_samples, test_data = one_experiment(experiment_key)

# save results
if not os.path.exists("results"):
    os.mkdir("results")

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
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
np.savez_compressed(
    datapath,
    final_params=final_params,
    best_params=best_params,
    true_A=A,
    true_beta=BETA,
    true_delta=DELTA,
    true_tau=TAU,
    loss_history=loss_history,
    replacement_rates=replacement_rates,
    test_xs=test_xs,
    test_data=test_data,
    test_samples=test_samples,
)

