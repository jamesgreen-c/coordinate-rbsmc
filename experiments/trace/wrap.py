"""Run TRACE inference, then reuse its posterior R and alpha in a second run.

Use the same options as experiment.py. The posterior-results option is supplied
by this wrapper automatically. Run from experiments/trace as usual.
"""
import subprocess
import sys
from pathlib import Path


RESULT_PREFIX = "Saved results to: "


def run_experiment(script, arguments):
    """
    Stream experiment output and return its reported result directory.

    experiment.py supplies the canonical directory name, including the kernel,
    s/b abbreviations and Ralpha mode; the wrapper never guesses kernel names.
    """
    command = [sys.executable, "-u", str(script), *arguments]
    result_dir = None
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith(RESULT_PREFIX):
                result_dir = Path(line[len(RESULT_PREFIX):].strip())
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    if result_dir is None or not (result_dir / "data.npz").is_file():
        raise RuntimeError("Experiment did not report a saved data.npz result.")
    return result_dir


def main():
    arguments = sys.argv[1:]
    if any(arg == "--posterior-results" or arg.startswith("--posterior-results=")
           for arg in arguments):
        raise SystemExit("wrap.py supplies --posterior-results automatically; omit it.")
    script = Path(__file__).resolve().with_name("experiment.py")
    if any(arg in ("-h", "--help") for arg in arguments):
        subprocess.run([sys.executable, str(script), *arguments], check=True)
        return

    print("Running inference for R and alpha", flush=True)
    inferred_dir = run_experiment(script, arguments)
    print(f"Running with R and alpha from {inferred_dir}", flush=True)
    run_experiment(script, [*arguments, "--posterior-results", str(inferred_dir)])


if __name__ == "__main__":
    main()
