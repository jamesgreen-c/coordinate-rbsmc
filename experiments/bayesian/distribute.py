# ARGS PARSING
import argparse
import os

from itertools import product
from rbsmc.utils.printing import ctext

parser = argparse.ArgumentParser()
parser.add_argument("--i", dest="i", type=int, default=-1)
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--N", dest="N", type=int, default=31)
parser.add_argument("--M", dest="M", type=int, default=1)
parser.add_argument("--burnin", dest="burnin", type=int, default=1000)
parser.add_argument("--samples", dest="samples", type=int, default=500)
parser.add_argument("--phi", dest="phi", type=float, default=0.1)
parser.add_argument("--log-var", dest="log_var", type=float, default=0)

args = parser.parse_args()


def results_exist(*, D, T, steps, args, kernel) -> bool:
    """Mirror experiment.py's experiment_name + datapath convention and check if results already exist."""

    if kernel == 0:
        kernel_name = "CSMC"
    elif kernel == 1:
        kernel_name = "RB_CSMC"
    elif kernel == 2:
        kernel_name = "GUEANT"
    else:
        raise ValueError("Invalid kernel int provided: must be in [0, 1, 2]")
    
    experiment_name = "kernel={},D={},T={},steps={},phi={},log-var={},N={},samples={},burnin={},conditional={},seed={}"
    experiment_name = experiment_name.format(
        kernel_name,
        D,
        T,
        steps,
        args.phi,
        args.log_var,
        args.N,
        args.samples,
        args.burnin,
        True,
        args.seed,
    )

    datapath = os.path.join("results", experiment_name, "data.npz")
    return os.path.exists(datapath)


DS = (3, 10, 15, 20)
TS = (500, 1000, 1500, 2000, 2500, 3000)
KERNELS = (0, 1, 2)

combination = [(D, T, kernel) for D, T, kernel in product(DS, TS, KERNELS) if D < 15 or T >= 1500][::-1]
print(f"Number of experiments: {len(combination)}")

if args.i != -1 and not (0 <= args.i < len(combination)):
    raise ValueError(f"--i must be in [0, {len(combination)-1}] or -1, got {args.i}")

indices = range(len(combination)) if args.i == -1 else [args.i]

for j in indices:
    D, T, kernel = combination[j]
    steps = T - 1

    if results_exist(D=D, T=T, steps=steps, args=args, kernel=kernel):
        print(ctext(f"Skipping (already run): kernel={kernel} D={D}, T={T}, steps={steps}, N={args.N}, samples={args.samples}, burnin={args.burnin}", "yellow"))
        continue

    exec_str = "python3 experiment.py --kernel {} --D {} --T {} --steps {} --N {} --M {} --samples {} --burnin {} --phi {} --log-var {} --seed {}"
    exec_str = exec_str.format(kernel, D, T, steps, args.N, args.M, args.samples, args.burnin, args.phi, args.log_var, args.seed)
    print("\nExecuting:", ctext(exec_str, "green"))
    # os.system(exec_str)