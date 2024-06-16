from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils


def _stack_from_df(df: pd.DataFrame, prefix: str, wxyz: str):
    cols = [prefix + ele for ele in wxyz]
    arr = []
    for col in cols:
        arr.append(df[col].to_numpy()[:, None])
        assert arr[-1].ndim == 2
    return np.concatenate(arr, axis=1)


@cache
def _load_data(trial: int):
    path = Path(__file__).parent.joinpath(f"dataset/trial{trial}")

    downloader = lambda file: path.joinpath(file)

    omc = pd.read_csv(downloader("omc.csv"), delimiter=",", skiprows=2)
    omc_hz = int(open(downloader("omc.csv")).readline().split(":")[1].lstrip().rstrip())
    imu_rigid = pd.read_csv(downloader("imu_rigid.csv"), delimiter=",", skiprows=2)
    imu_rigid_hz = int(
        open(downloader("imu_rigid.csv")).readline().split(":")[1].lstrip().rstrip()
    )
    imu_nonrigid = pd.read_csv(
        downloader("imu_nonrigid.csv"), delimiter=",", skiprows=2
    )
    imu_nonrigid_hz = int(
        open(downloader("imu_nonrigid.csv")).readline().split(":")[1].lstrip().rstrip()
    )
    assert imu_rigid_hz == imu_nonrigid_hz

    data = {}
    for seg in range(1, 6):
        data_seg = {}
        seg = f"seg{seg}"
        data[seg] = data_seg

        # quat
        data_seg["quat"] = _stack_from_df(omc, seg + "_quat_", "wxyz")

        # markers
        for marker in range(1, 5):
            marker = f"marker{marker}"
            data_seg[marker] = _stack_from_df(omc, seg + "_" + marker + "_", "xyz")

        # imu
        for imu_name, imu in zip(
            ["imu_rigid", "imu_nonrigid"], [imu_rigid, imu_nonrigid]
        ):
            data_seg_imu = {}
            data_seg[imu_name] = data_seg_imu
            for accgyrmag in ["acc", "gyr", "mag"]:
                data_seg_imu[accgyrmag] = _stack_from_df(
                    imu, seg + "_" + accgyrmag + "_", "xyz"
                )

    return data, omc_hz, imu_rigid_hz


def load_data(
    trial: int,
    resample_to_hz: float = 100.0,
) -> dict:
    assert trial in [1, 2]

    data, hz_omc, hz_imu = _load_data(trial)

    data = utils.resample(
        data,
        hz_in=utils.hz_helper(
            data.keys(),
            imus=["imu_rigid", "imu_nonrigid"],
            hz_imu=hz_imu,
            hz_omc=hz_omc,
        ),
        hz_out=resample_to_hz,
        vecinterp_method="cubic",
    )
    data = utils.crop_tail(data, resample_to_hz, strict=True, verbose=False)

    return data
