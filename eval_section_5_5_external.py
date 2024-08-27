from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import qmt
import ring
from ring import maths

root = Path(__file__).parent


def metric(q, qhat, hz):
    phi = maths.angle_error(q, qhat)
    return np.mean(np.rad2deg(phi[int(5 * hz) :]))  # noqa: E203


ringnet = ring.RING(lam=None, Ts=None)


class AG(NamedTuple):
    acc: np.ndarray
    gyr: np.ndarray
    q: np.ndarray


def predict(hz, *pairs: tuple[AG]):
    N = len(pairs)
    T = pairs[0].acc.shape[0]
    X = np.zeros((T, N, 10))
    X[..., -1] = 1 / hz
    for i in range(N):
        X[:, i, :3] = pairs[i].acc
        X[:, i, 3:6] = pairs[i].gyr
    qhat, _ = ringnet.apply(X=X, lam=tuple(range(-1, N - 1)))
    errors = [metric(p.q, qhat[:, i], hz) for i, p in enumerate(pairs)]
    return qhat, errors


def load_data_repoIMU_pendulum(id, seg: int):
    p = root.joinpath(
        f"data/repoIMU_dataset/Pendulum_Test0{id}_Trial1_Segment_{seg}.csv"
    )
    df = pd.read_csv(p, skiprows=1, delimiter=";")

    # earth-omc to seg
    q = df.iloc[:, list(range(1, 5))].to_numpy().astype(float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)

    def load(i: int):
        return df.iloc[:, list(range(i, i + 3))].to_numpy().astype(float)

    acc, gyr, mag = load(5), load(8), load(11)

    return q, acc, gyr, mag


def get_acc_gyr_openaxes(sce, imu):
    df = pd.read_csv(
        root.joinpath(f"data/openaxes_dataset/synced_imu/{sce}/{imu}_adx.csv")
    )
    acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyr = df[["gyr_x", "gyr_y", "gyr_z"]].to_numpy()
    return acc, gyr


def get_gt_openaxes(sce, imu):
    return pd.read_csv(
        root.joinpath(f"data/openaxes_dataset/groundtruth/robot-{sce}/{imu}.csv")
    )[[f"QUATERNION_{i}" for i in range(4)]].to_numpy()


# -- RepoIMU --#
hz = 90
T_repoimu = (
    load_data_repoIMU_pendulum("2a", 1)[0].shape[0]
    + load_data_repoIMU_pendulum("2b", 1)[0].shape[0]
) / hz
print(f"RepoIMU contains T={T_repoimu} seconds")

e_rel = []
for id in ["2a", "2b"]:
    q1, acc1, gyr1, _ = load_data_repoIMU_pendulum(id, 1)
    q2, acc2, gyr2, _ = load_data_repoIMU_pendulum(id, 2)
    _, errors = predict(
        hz, AG(acc1, gyr1, q1), AG(acc2, gyr2, maths.quat_mul(maths.quat_inv(q1), q2))
    )
    e_rel.append(errors[1])

print(
    "RMAE [deg] for RepoIMU (groundtruth using OMC): ",
    np.mean(e_rel),
    "+/-",
    np.std(e_rel),
)

# -- OpenAXES --#
hz = 125

# exclude the trials with
# - scenario = circle-200
# - scenario = triangle-100
# because the time synchronization failed (!). The imus and ground truth data stream
# have no consistent time relation to each other
valid_scenarios = [
    "circle-100",
    "circle-150",
    "circle-250",
    "triangle-150",
    "triangle-200",
    "triangle-250",
]

T_openaxes = (
    sum([get_gt_openaxes(sce, "radius").shape[0] for sce in valid_scenarios]) / hz
)
print(f"OpenAXES contains T={T_openaxes} seconds")
errors = []

for sce in valid_scenarios:
    acc_rad, gyr_rad = get_acc_gyr_openaxes(sce, "radius")
    acc_car, gyr_car = get_acc_gyr_openaxes(sce, "carpus")
    q_rad = get_gt_openaxes(sce, "radius")
    q_car = get_gt_openaxes(sce, "carpus")
    q = qmt.qmult(qmt.qinv(q_car), q_rad)

    # ignore those errors here; the alignment of IMUs and segments are not given
    # and so the groundtruth axis is expressed in an unknown frame; instead, use
    # the joint axis angle as ground truth because the serial robot uses perfect
    # hinge joints anyways
    qhat, _ = predict(hz, AG(acc_rad, gyr_rad, q_rad), AG(acc_car, gyr_car, q_car))

    # exclude 5 second of warmup time
    warmup = hz * 5
    error = np.rad2deg(
        np.mean(np.abs(qmt.quatAngle(qhat[warmup:, 1]) - qmt.quatAngle(q[warmup:])))
    )

    errors.append(error)

print(
    "RMAE [deg] for OpenAXES IMUs with accelerometer ADXL355:",
    np.mean(errors),
    "+/-",
    np.std(errors),
)
