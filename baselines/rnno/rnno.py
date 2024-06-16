from pathlib import Path

import ring

folder = Path(__file__).parent


def _params(v1_or_v2: str, xy_or_yz: str):
    return folder.joinpath(f"rnno_{v1_or_v2}_{xy_or_yz}.pickle")


RNNO_v1_xy = ring.ml.RNNO(
    12,
    return_quats=1,
    params=_params("v1", "xy"),
    eval=False,
    v1=1,
    rnn_layers=(400, 300),
    linear_layers=(200, 100, 50, 50, 25, 25),
).unwrapped
RNNO_v1_yz = ring.ml.RNNO(
    12,
    return_quats=1,
    params=_params("v1", "yz"),
    eval=False,
    v1=1,
    rnn_layers=(400, 300),
    linear_layers=(200, 100, 50, 50, 25, 25),
).unwrapped
# these pickle files correspond to the runs with ids that end with `...170d`
RNNO_v2_xy = ring.ml.RNNO(
    12, return_quats=1, params=_params("v2", "xy"), eval=False, hidden_state_dim=200
).unwrapped
RNNO_v2_yz = ring.ml.RNNO(
    12, return_quats=1, params=_params("v2", "yz"), eval=False, hidden_state_dim=200
).unwrapped
