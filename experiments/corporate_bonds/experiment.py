import argparse
import os
import time

import jax
import jax.numpy as jnp

import numpy as np
import tqdm

from experiments.corporate_bonds.kernels import KernelType, get_csmc_kernel
from experiments.corporate_bonds.model import get_data

from rbsmc.utils.common import force_move, barker_move
from rbsmc.utils.resamplings import killing, multinomial

# jax.config.update("jax_enable_x64", False)
# jax.config.update("jax_platform_name", "cpu")

# ARGS PARSING
parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=10)
parser.add_argument("--D", dest="D", type=int, default=1)
parser.add_argument("--K", dest="K", type=int, default=1)
parser.add_argument("--M", dest="M", type=int, default=5)
parser.add_argument("--steps", type=int, default=100)

parser.add_argument("--independent", action="store_true")
parser.set_defaults(independent=False)

parser.add_argument("--log-var", dest="log_var", type=float, default=0)
parser.add_argument("--phi", dest="phi", type=float, default=0.8)


parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--style", dest="style", type=str, default="guided")

parser.add_argument("--backward", action='store_true')
parser.add_argument('--no-backward', dest='backward', action='store_false')
parser.set_defaults(backward=True)

parser.add_argument("--resampling", dest='resampling', type=str, default="multinomial")
parser.add_argument("--last-step", dest='last_step', type=str, default="barker")
parser.add_argument("--N", dest="N", type=int, default=31)  # total number of particles is N + 1

parser.add_argument("--debug", action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

args = parser.parse_args()
kernel_type = KernelType.GUEANT

print(f"""
##################################
#  CORPORATE BOND EXPERIMENT     #
##################################
Configuration:
    - T:         {args.T}
    - kernel:    {kernel_type.name}
    - style:     {args.style}
    - D:         {args.D}
    - M:         {args.M}
    - steps:     {args.steps}
""")

# PARAMETERS
KEY = jax.random.PRNGKey(args.seed)
ALL_KEYS = jax.random.split(KEY, args.K + 1)
WARMUP_KEY = ALL_KEYS[0]
EXPERIMENT_KEYS = ALL_KEYS[1:]

if args.resampling == "killing":
    resampling_fn = killing
elif args.resampling == "multinomial":
    resampling_fn = multinomial
else:
    raise ValueError(f"Unknown resampling {args.resampling}")

if args.last_step == "forced":
    last_step_fn = force_move
elif args.last_step == "barker":
    last_step_fn = barker_move
else:
    raise ValueError(f"Unknown last step {args.last_step}")

# --- dynamics config ---
A = args.phi * jnp.eye(args.D)
CHOL_Q0 = 0.1 * jnp.eye(args.D)
CHOL_H0 = 0.1 * jnp.eye(args.D)
CHOL_Q = 10 ** (args.log_var / 2) * jnp.eye(args.D)  # independent spreads

vol_eta = 0.10 * jnp.array([1.00, 1.24, 1.38])
corr_eta = jnp.array([
    [1.000, 0.60, 0.58],
    [0.60, 1.000, 0.65],
    [0.58, 0.65, 1.000],
])
Q_eta = corr_eta * vol_eta[:, None] * vol_eta[None, :]
CHOL_H_TRUE = jnp.linalg.cholesky(Q_eta)

if args.independent:
    CHOL_H = 0.1 * jnp.eye(args.D)
else:
    CHOL_H = CHOL_H_TRUE

CHOL_R = 0.1 * jnp.eye(args.D)
PSI = 0.05 * jnp.ones(args.D)
ALPHA = 0.10 * jnp.ones(args.D)

DTs = jnp.repeat(args.T / args.steps, args.steps)

# ------- experiment function -------

@(jax.jit if not args.debug else lambda x: x)
def one_experiment(key):
    data_key, sample_key = jax.random.split(key)

    true_xs, (ys, indices, obs_types), *_ = get_data(
        data_key, args.D, DTs, 
        A, PSI, CHOL_Q0, CHOL_Q, CHOL_H0, CHOL_H_TRUE, CHOL_R, ALPHA,
        sparsity_factor=10.0
    )

    csmc_kernel, csmc_init, *_ = get_csmc_kernel(
        ys, indices, obs_types, 
        ALPHA, PSI, 
        A, CHOL_Q0, CHOL_Q, CHOL_H0, CHOL_H, CHOL_R,
        N=args.N, dts=DTs,
        resampling_func=resampling_fn,
        backward=args.backward,
        ancestor_move_func=last_step_fn,
        style=args.style, 
        conditional=False
    )
    init_state = csmc_init(true_xs)   # no leakage as conditional = False

    def _independent_sample(k_):
        return csmc_kernel(k_, init_state)

    sample_keys = jax.random.split(sample_key, args.M)
    samples, *_ = jax.vmap(_independent_sample)(sample_keys)

    return samples, true_xs, (ys, indices, obs_types)

# Compile once, without executing a full experiment
start = time.time()
compiled_one_experiment = one_experiment.lower(WARMUP_KEY).compile()
print(f"Compile time: {time.time() - start:.2f} seconds.")

# storage
zs_all = np.empty((args.K, args.M, args.steps, args.D))
etas_all = np.empty((args.K, args.M, args.steps, args.D))

true_zs_all = np.empty((args.K, args.steps, args.D))
true_etas_all = np.empty((args.K, args.steps, args.D))

bond_indices_all = np.empty((args.K, args.steps))
event_types_all = np.empty((args.K, args.steps))
alphas_all = np.empty((args.K, args.steps))
obs_values_all = np.empty((args.K, args.steps))

for k, key_k in enumerate(tqdm.tqdm(EXPERIMENT_KEYS, desc="Experiment: ")):
    samples_k, true_xs_k, obs_k = compiled_one_experiment(key_k)

    zs, etas = samples_k
    true_zs, true_etas = true_xs_k
    ys_k, indices_k, obs_types_k = obs_k

    zs_all[k] = zs
    etas_all[k] = etas
    true_zs_all[k] = true_zs
    true_etas_all[k] = true_etas
    bond_indices_all[k] = indices_k
    event_types_all[k] = obs_types_k
    obs_values_all[k] = ys_k

if not os.path.exists("results"):
    os.mkdir("results")

experiment_name = "kernel={},style={},D={},T={},N={},steps={},M={},independent={},seed={}"
experiment_name = experiment_name.format(
    kernel_type.name,
    args.style,
    args.D,
    args.T,
    args.N,
    args.steps,
    args.M,
    args.independent,
    args.seed,
)

dirpath = f"results/{experiment_name}"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

datapath = f"{dirpath}/data.npz"
np.savez_compressed(
    datapath,
    zs=zs_all,
    etas=etas_all,
    true_zs=true_zs_all,
    true_etas=true_etas_all,
    bond_indices=bond_indices_all,
    event_types=event_types_all,
    alphas=alphas_all,
    obs_values=obs_values_all
)