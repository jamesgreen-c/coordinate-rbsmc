import argparse
import os
import shlex
import subprocess

from rbsmc.utils.printing import ctext
from experiments.corporate_bonds.kernels import KernelType


parser = argparse.ArgumentParser()

parser.add_argument("--T", dest="T", type=int, default=1)
parser.add_argument("--K", dest="K", type=int, default=1)
parser.add_argument("--M", dest="M", type=int, default=5)

parser.add_argument("--independent", action="store_true")
parser.set_defaults(independent=False)

parser.add_argument("--log-var", dest="log_var", type=float, default=0)
parser.add_argument("--phi", dest="phi", type=float, default=0.8)

parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--style", dest="style", type=str, default="bootstrap")

parser.add_argument("--backward", action="store_true")
parser.add_argument("--no-backward", dest="backward", action="store_false")
parser.set_defaults(backward=True)

parser.add_argument("--resampling", dest="resampling", type=str, default="multinomial")
parser.add_argument("--last-step", dest="last_step", type=str, default="barker")
parser.add_argument("--N", dest="N", type=int, default=31)

parser.add_argument("--debug", action="store_true")
parser.add_argument("--no-debug", dest="debug", action="store_false")
parser.set_defaults(debug=False)

parser.add_argument("--i", dest="i", type=int, default=-1)
parser.add_argument("--start-i", dest="start_i", type=int, default=0)

parser.add_argument("--run", dest="run", action="store_true")
parser.add_argument("--no-run", dest="run", action="store_false")
parser.set_defaults(run=False)

args = parser.parse_args()


# ---------------------------------------------------------------------
# Sweep controls
# ---------------------------------------------------------------------
KERNEL_TYPE = KernelType.CSMC

DS = (1, 5, 10, 25, 50, 75, 100)
BASE_T = args.T


def experiment_id(*, D, T, steps):
    return {
        "D": int(D),
        "T": int(T),
        "steps": int(steps),
        "kernel": KERNEL_TYPE,
        "style": args.style,
    }


def results_exist(*, kernel, style, D, T, steps, args) -> bool:
    """
    Mirrors experiment.py's experiment_name + datapath convention.
    Must match experiment.py exactly.
    """
    experiment_name = "kernel={},style={},D={},T={},N={},steps={},M={},independent={},seed={}"
    experiment_name = experiment_name.format(
        kernel.name,
        style,
        D,
        T,
        args.N,
        steps,
        args.M,
        args.independent,
        args.seed,
    )

    datapath = os.path.join("results", experiment_name, "data.npz")
    return os.path.exists(datapath)


def build_combinations():
    combinations = []

    for D in DS:
        steps = 10 * D
        combinations.append(experiment_id(D=D, T=BASE_T, steps=steps))

    return combinations


def build_command(*, combo, args):
    cmd = [
        "python3", "experiment.py",
        "--style", combo["style"],
        "--D", str(combo["D"]),
        "--T", str(combo["T"]),
        "--steps", str(combo["steps"]),
        "--K", str(args.K),
        "--N", str(args.N),
        "--M", str(args.M),
        "--log-var", str(args.log_var),
        "--phi", str(args.phi),
        "--resampling", args.resampling,
        "--last-step", args.last_step,
        "--seed", str(args.seed),
    ]

    if args.independent:
        cmd.append("--independent")

    if not args.backward:
        cmd.append("--no-backward")

    if args.debug:
        cmd.append("--debug")

    return cmd


COMBINATIONS = build_combinations()

print(f"Number of experiments: {len(COMBINATIONS)}")
print(f"T:                     {BASE_T}")
print(f"D grid:                {DS}")
print(f"steps rule:            steps = 10 * D")
print(f"steps grid:            {[c['steps'] for c in COMBINATIONS]}")
print(f"kernel:                {KERNEL_TYPE.name}")
print(f"style:                 {args.style}")
print(f"N:                     {args.N}")
print(f"M:                     {args.M}")
print(f"K:                     {args.K}")
print(f"independent:           {args.independent}")
print(f"backward:              {args.backward}")
print(f"resampling:            {args.resampling}")
print(f"last step:             {args.last_step}")

if args.i != -1 and not (0 <= args.i < len(COMBINATIONS)):
    raise ValueError(f"--i must be in [0, {len(COMBINATIONS) - 1}] or -1, got {args.i}")

if not (0 <= args.start_i < len(COMBINATIONS)):
    raise ValueError(f"--start-i must be in [0, {len(COMBINATIONS) - 1}], got {args.start_i}")

indices = range(args.start_i, len(COMBINATIONS)) if args.i == -1 else [args.i]

for j in indices:
    combo = COMBINATIONS[j]

    D = combo["D"]
    T = combo["T"]
    steps = combo["steps"]
    kernel = combo["kernel"]
    style = combo["style"]

    if results_exist(kernel=kernel, style=style, D=D, T=T, steps=steps, args=args):
        print(
            ctext(
                f"Skipping already run: kernel={kernel.name}, style={style}, T={T}, D={D}, steps={steps}, N={args.N}, M={args.M}, K={args.K}, independent={args.independent}",
                "yellow",
            )
        )
        continue

    cmd = build_command(combo=combo, args=args)
    exec_str = shlex.join(cmd)

    print("\nExecuting:", ctext(f"[{j}/{len(COMBINATIONS) - 1}] D={D}, T={T}, steps={steps} :: {exec_str}", "green"))

    if args.run:
        subprocess.run(cmd, check=True)