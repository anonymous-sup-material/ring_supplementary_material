import os
from pathlib import Path

import numpy as np
import pandas as pd
import qmt
import scipy.interpolate as interp
import scipy.optimize as opt

# 1. Donwload the repository
# 2. Use the .ipynb to use VQF or Madgwick to estimate orientations from IMUs
# 3. Use this to sync the IMU data to the groundtruth encoder data by comparing the
#   - orientation from IMUs
#   - orientation from encoders
PATH_TO_REPO = os.path.expanduser("~/Downloads/openaxes-example-robot-dataset")
OUTPUT_PATH = os.path.expanduser("~/Download/synced_imu")


def csv_dirname(scenario, method=""):
    data_dir = os.path.join(PATH_TO_REPO, "measure_raw-2022-09-15")

    if "robot" in method:
        return os.path.join(data_dir, "robot", "robot-" + scenario)
    imu_dir = os.path.join(data_dir, "imu-" + scenario)
    if method == "":
        return imu_dir
    return os.path.join(imu_dir, method)


def resampleSignal(signal, time, scale, offset, return_time=False):
    duration = time[-1] - time[0]
    t2 = np.linspace(0, duration, len(signal))
    t2_resample_shift = (time - time[0] - offset) / scale
    resampled = interp.interpn(
        (t2,), signal, t2_resample_shift, bounds_error=False, fill_value=signal[0]
    )
    if return_time:
        return resampled, t2_resample_shift
    return resampled


def alignSequences(s1, s2, t1, t2=None, offset_only=False, guess_initial=False):
    """
    Scale and offset signal s2 so that its difference to s1 is minimal.
    The returned `scale` and `offset` must be applied relative to the start of `t1`:
    `aligned_t2 = t1[0] + np.linspace(0, t1[-1] - t1[0], len(s2)) * scale + offset

    Parameters
    ----------

    `offset_only` : boolean
        Disable scaling, only shift the signal.
    `guess_initial` : boolean
        Look at the peaks of the signals for initial scale and offset guesses.

    Return
    ------
    tuple (`scale`, `offset`, `cost`)
    """
    t1 = t1 - t1[0]  # Shift time origin to t=0
    duration = t1[-1] - t1[0]
    if t2 is None:
        t2 = np.linspace(0, duration, len(s2))
    assert len(t1) == len(s1), "s1 and t1 must have the same length"
    assert len(t2) == len(s2), "s2 and t2 must have the same length"

    def resample_s2(scale, offset):
        return interp.interpn(
            (t2,),
            s2,
            (t1 - offset * duration) / scale,
            bounds_error=False,
            fill_value=s2[0],
        )

    def costFunction(scale_offset):
        residuals = s1 - resample_s2(*scale_offset)
        return residuals  # for methods that need a single residual value

    scale, offset = 1, 0

    if guess_initial:
        from scipy.signal import find_peaks

        amplitude = np.max(s1) - np.min(s1)

        def find_peak_indices(x: np.ndarray):
            indices, _props = find_peaks(x, prominence=0.3 * amplitude)
            return indices

        # Estimate initial alignment using peak positions
        indices1 = find_peak_indices(s1)
        indices2 = find_peak_indices(s2)
        if len(indices1) != len(indices2):
            # print(indices1, ' != ', indices2)
            indices1 = find_peak_indices(-s1)
            indices2 = find_peak_indices(-s2)
        if len(indices1) == len(indices2):
            peaks_t1, peaks_t2 = t1[indices1], t2[indices2]
            a = np.column_stack([peaks_t2, np.ones(len(peaks_t2)) * duration])
            scale_offset, residuals, rank, singular = np.linalg.lstsq(
                a, peaks_t1, rcond=None
            )
            scale, offset = scale_offset

    if offset_only:
        result = opt.least_squares(
            lambda ofs: costFunction((1.0, ofs)), [offset], bounds=([-0.5], [0.5])
        )
        return 1.0, result.x[0] * duration, result.cost
    else:
        result = opt.least_squares(
            costFunction, [scale, offset], bounds=([0.5, -0.5], [2, 0.5])
        )
        return result.x[0], result.x[1] * duration, result.cost


def imu_csv(scenario, imu):
    return pd.read_csv(os.path.join(csv_dirname(scenario, "vqf-cal"), f"{imu}.csv"))


def get_imu_heading(scenario, imu):

    quat = imu_csv(scenario, imu)[
        ["QUATERNION_0", "QUATERNION_1", "QUATERNION_2", "QUATERNION_3"]
    ].to_numpy()
    return qmt.eulerAngles(quat, axes="xyz")[:, 1]


def get_robot_time_heading(scenario):

    time_q0_q5 = pd.read_csv(csv_dirname(scenario, "robot") + ".csv")[
        ["timestamp", "q0", "q1", "q2", "q3", "q4", "q5"]
    ].to_numpy()
    robot_heading = -time_q0_q5[:, 1]
    time = np.array(time_q0_q5[:, 0] * 1e6, dtype=int)
    time = time - time[0]
    return time, robot_heading


def save_imu(filename, acc, gyr):
    filename = Path(OUTPUT_PATH).joinpath(filename)
    filename.parent.mkdir(exist_ok=True, parents=True)
    unpack_3D = lambda d: {
        key + "_" + "xyz"[i]: val[:, i] for key, val in d.items() for i in range(3)
    }
    pd.DataFrame(unpack_3D(dict(acc=acc, gyr=gyr))).to_csv(filename)


scenarios = [
    "circle-100",
    "circle-150",
    "circle-200",
    "circle-250",
]  # "triangle-100", "triangle-150", "triangle-200", "triangle-250"]
for scenario in scenarios:
    for imu in ["carpus", "radius"]:
        time, robot_heading = get_robot_time_heading(scenario)
        imu_heading = get_imu_heading(scenario, imu)
        scale, offset, cost = alignSequences(robot_heading, imu_heading, time)
        print(scenario, imu, scale, offset)

        df_imu = imu_csv(scenario, imu)
        acc_adx = df_imu[[f"ACCEL_ADXL355_{i}" for i in range(3)]].to_numpy()
        acc_bmi = df_imu[[f"ACCEL_BMI160_{i}" for i in range(3)]].to_numpy()
        gyr = df_imu[[f"GYRO_{i}" for i in range(3)]].to_numpy()

        resample = lambda a: np.concatenate(
            [resampleSignal(a[:, i], time, scale, offset)[:, None] for i in range(3)],
            axis=-1,
        )
        save_imu(f"{scenario}/{imu}_adx.csv", resample(acc_adx), resample(gyr))
        save_imu(f"{scenario}/{imu}_bmi.csv", resample(acc_bmi), resample(gyr))
