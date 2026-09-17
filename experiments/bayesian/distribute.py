# ARGS PARSING
import argparse
import os
import shlex
import subprocess

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

parser.add_argument("--full-inference", action="store_true")
parser.add_argument("--no-full-inference", dest="full_inference", action="store_false")
parser.set_defaults(full_inference=False)

args = parser.parse_args()

KERNEL_NAMES = {
    0: "CSMC",
    1: "RB_CSMC",
    2: "GUEANT",
}

BACKWARD_MODES = {
    0: "ffbsi",
    1: "reduced",
    2: "ffbsi",
}


def retention_config(D: int) -> tuple[int, int | None]:
    """Return memory-safe history settings for a given state dimension."""
    if D >= 100:
        return 10, 20
    if D >= 50:
        return 5, 20
    return 1, None


def results_exist(*, D, T, steps, args, kernel, thin, saved_paths) -> bool:
    """Mirror experiment.py's experiment_name + datapath convention and check if results already exist."""
    if kernel not in KERNEL_NAMES:
        raise ValueError("Invalid kernel int provided: must be in [0, 1, 2]")

    experiment_name = "kernel={},D={},T={},steps={},phi={},N={},samples={},burnin={},full-inference={},conditional={},seed={},backward-mode={}"
    experiment_name = experiment_name.format(
        KERNEL_NAMES[kernel],
        D,
        T,
        steps,
        args.phi,
        args.N,
        args.samples,
        args.burnin,
        args.full_inference,
        True,
        args.seed,
        BACKWARD_MODES[kernel],
    )

    datapath = os.path.join("results", experiment_name, "data.npz")
    return os.path.exists(datapath)


DS = (3, 10, 15, 20, 50, 100)
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
    thin, saved_paths = retention_config(D)

    if results_exist(
        D=D,
        T=T,
        steps=steps,
        args=args,
        kernel=kernel,
        thin=thin,
        saved_paths=saved_paths,
    ):
        print(ctext(f"Skipping (already run): kernel={KERNEL_NAMES[kernel]}, backward-mode={BACKWARD_MODES[kernel]}, D={D}, T={T}, steps={steps}, N={args.N}, samples={args.samples}, burnin={args.burnin}, full-inference={args.full_inference}", "yellow"))
        continue

    inference_flag = "--full-inference" if args.full_inference else "--no-full-inference"
    command = [
        "python3",
        "experiment.py",
        "--kernel", str(kernel),
        "--D", str(D),
        "--T", str(T),
        "--steps", str(steps),
        "--N", str(args.N),
        "--M", str(args.M),
        "--samples", str(args.samples),
        "--burnin", str(args.burnin),
        "--phi", str(args.phi),
        "--seed", str(args.seed),
        "--backward-mode", BACKWARD_MODES[kernel],
        "--thin", str(thin),
        inference_flag,
    ]
    if saved_paths is not None:
        command.extend(("--saved-paths", str(saved_paths)))

    printable_command = " ".join(shlex.quote(part) for part in command)
    print("\nExecuting:", ctext(printable_command, "green"))
    # subprocess.run(command, check=True)
