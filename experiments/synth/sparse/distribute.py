import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from itertools import product

from rbsmc.utils.printing import ctext


parser = argparse.ArgumentParser()
parser.add_argument("--i", dest="i", type=int, default=-1)
parser.add_argument("--seed", dest="seed", type=int, default=1234)
parser.add_argument("--N", dest="N", type=int, default=31)
parser.add_argument("--M", dest="M", type=int, default=3)
parser.add_argument("--burnin", dest="burnin", type=int, default=1000)
parser.add_argument("--samples", dest="samples", type=int, default=500)
parser.add_argument("--theta-repeat", type=int, default=30)
parser.add_argument("--phi", dest="phi", type=float, default=0.1)
parser.add_argument("--analysis", action="store_true")
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

ROOT = Path(__file__).resolve().parent
SHARED_ROOT = Path(__file__).resolve().parents[1]


def retention_config(D: int) -> tuple[int, int | None]:
    """Return memory-safe history settings for a given state dimension."""
    return 20, 20

def results_exist(*, D, T, steps, args, kernel, thin, saved_paths) -> bool:
    """Check the result path used by the shared experiment script."""
    if kernel not in KERNEL_NAMES:
        raise ValueError("Invalid kernel int provided: must be in [0, 1, 2]")

    experiment_name = "kernel={},D={},T={},steps={},N={},samples={},burnin={},seed={},backward={}"
    experiment_name = experiment_name.format(
        KERNEL_NAMES[kernel],
        D,
        T,
        steps,
        args.N,
        args.samples,
        args.burnin,
        args.seed,
        BACKWARD_MODES[kernel],
    )

    datapath = ROOT / "results" / experiment_name / "data.npz"
    return datapath.is_file()

def run_command(command):
    printable_command = " ".join(shlex.quote(part) for part in command)
    print("\nExecuting:", ctext(printable_command, "green"))
    subprocess.run(command, check=True)


DS = (3, 10, 15, 20, 50)
TS = (711,)
KERNELS = (0, 1, 2)

combination = [(D, T, kernel) for D, T, kernel in product(DS, TS, KERNELS)][::-1]
print(f"Number of experiments: {len(combination)}")

if args.i != -1 and not (0 <= args.i < len(combination)):
    raise ValueError(f"--i must be in [0, {len(combination)-1}] or -1, got {args.i}")

indices = range(len(combination)) if args.i == -1 else [args.i]

for j in indices:
    D, T, kernel = combination[j]
    steps = (T - 1) * D // 3  # roughly one observation per bond every 3 days
    thin, saved_paths = retention_config(D)

    inference_flag = "--full-inference" if args.full_inference else "--no-full-inference"
    common_args = [
        "--D", str(D),
        "--T", str(T),
        "--steps", str(steps),
        "--N", str(args.N),
        "--samples", str(args.samples),
        "--burnin", str(args.burnin),
        "--phi", str(args.phi),
        "--seed", str(args.seed),
        "--backward-mode", BACKWARD_MODES[kernel],
        "--thin", str(thin),
        "--root", str(ROOT),
        inference_flag,
    ]
    if saved_paths is not None:
        common_args.extend(("--saved-paths", str(saved_paths)))

    if results_exist(D=D, T=T, steps=steps, args=args, kernel=kernel, thin=thin, saved_paths=saved_paths):
        print(ctext(
            f"Skipping (already run): kernel={KERNEL_NAMES[kernel]}, "
            f"backward-mode={BACKWARD_MODES[kernel]}, D={D}, T={T}, "
            f"steps={steps}, N={args.N}, samples={args.samples}, "
            f"burnin={args.burnin}, full-inference={args.full_inference}",
            "yellow",
        ))
    else:
        command = [
            sys.executable,
            str(SHARED_ROOT / "experiment.py"),
            "--kernel", str(kernel),
            "--M", str(args.M),
            "--theta-repeat", str(args.theta_repeat),
            *common_args,
        ]
        run_command(command)

    if args.analysis:
        command = [
            sys.executable,
            str(SHARED_ROOT / "analysis.py"),
            "--kernel", KERNEL_NAMES[kernel],
            *common_args,
        ]
        run_command(command)
