import pandas as pd
from experiments.trace.dataset import TraceDataset


def read_csv(path, columns=None):
    """read only selected fields as trimmed strings, preserving leading zeros."""
    frame = pd.read_csv(
        path, 
        usecols=columns, 
        dtype="string", 
        keep_default_na=False,
        na_values=["", ".", "NA", "N/A", "NULL", "NaN"],
    )

    for column in frame.columns:
        frame[column] = frame[column].str.strip().replace("", pd.NA)
    return frame


# def get_data(path):
    
#     data = read_csv(
#         path, 
#         columns=["bond_idx", "yield-to-benchmark", "event_type", "dt"]
#     )
    
#     obs_values = data["yield-to-benchmark"].astype(float).to_numpy()
#     bond_idxs = data["bond_idx"].astype(int).to_numpy()
#     event_types = data["event_type"].astype(int).to_numpy()
#     dts = data["dt"].astype(float).to_numpy()
#     D = bond_idxs.max() + 1

#     return TraceDataset(
#         D=D,
#         dts=dts[1:],        # remove first dt=0
#         data=(obs_values, bond_idxs, event_types),
#     )

def get_data(path, n_bonds: int | None = None):

    data = read_csv(
        path,
        columns=["bond_idx", "yield-to-benchmark", "event_type", "dt"],
    )
    data["bond_idx"] = data["bond_idx"].astype(int)
    times = data["dt"].astype(float).cumsum()

    if n_bonds is not None:
        data = data[data["bond_idx"].isin(range(n_bonds))]

    obs_values = data["yield-to-benchmark"].astype(float).to_numpy()
    bond_idxs = data["bond_idx"].to_numpy()
    event_types = data["event_type"].astype(int).to_numpy()
    dts = times.loc[data.index].diff().iloc[1:].to_numpy()
    D = bond_idxs.max() + 1

    return TraceDataset(
        D=D,
        dts=dts,
        data=(obs_values, bond_idxs, event_types),
    )