import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--saved-paths", type=int, default=20)
parser.add_argument("--experiment", type=Path, required=True)
args = parser.parse_args()

if args.saved_paths < 1:
    parser.error("--saved-paths must be at least 1.")
keep = args.saved_paths

path = args.experiment.expanduser().resolve()
if path.suffix != ".npz":
    path = path / "data.npz"

if not path.is_file():
    raise FileNotFoundError(path)

answer = input(
    f"Overwrite {path}, keeping only the latest {keep} paths per chain? [y/n]: "
).strip().lower()
if answer != "y":
    print("Cancelled.")
    raise SystemExit(0)

with np.load(path, allow_pickle=True) as saved:
    data = {key: saved[key] for key in saved.files}

single_chain = np.asarray(data["params"].item()["H"]).ndim == 3
path_axis = 0 if single_chain else 1


def truncate(value):
    if value is None:
        return None
    value = np.asarray(value)
    index = [slice(None)] * value.ndim
    index[path_axis] = slice(-keep, None)
    return value[tuple(index)]


if "references" in data:
    reference = data["references"].item()
    if isinstance(reference, dict):
        fields = reference
    elif hasattr(reference, "_asdict"):
        fields = reference._asdict()
    else:
        fields = vars(reference)

    updates = {
        "trajectory": tuple(truncate(x) for x in fields["trajectory"])
    }
    for key in ("ancestors", "particle_indices"):
        if key in fields:
            updates[key] = truncate(fields[key])

    if isinstance(reference, dict):
        reference.update(updates)
    elif hasattr(reference, "_replace"):
        reference = reference._replace(**updates)
    elif hasattr(reference, "replace"):
        reference = reference.replace(**updates)
    else:
        for key, value in updates.items():
            setattr(reference, key, value)

    data["references"] = reference

if "trajectories" in data:
    data["trajectories"] = tuple(
        truncate(x) for x in data["trajectories"]
    )

for key in ("reference_particle_indices", "ancestors"):
    if key in data:
        data[key] = truncate(data[key])

config = data["config"].item() if "config" in data else {}
if isinstance(config, dict):
    config["saved_paths"] = keep
elif hasattr(config, "_replace"):
    config = config._replace(saved_paths=keep)
elif hasattr(config, "replace"):
    config = config.replace(saved_paths=keep)
else:
    config.saved_paths = keep
data["config"] = config

temporary = path.with_name(f"{path.stem}_truncated.tmp.npz")
np.savez_compressed(temporary, **data)
temporary.replace(path)

print(f"Saved latest keep paths per chain to {path}")