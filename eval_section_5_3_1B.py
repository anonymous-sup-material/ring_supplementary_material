import numpy as np
import ring

import baselines
from data import benchmark
from data import IMTP

_LPF_CUTOFF_FREQ = 5.0
methods = [
    ring.ml.base.LPF_FilterWrapper(
        baselines.TwoSeg1D("1d_corr", ollson=False), _LPF_CUTOFF_FREQ, None, quiet=True
    ),
    ring.ml.base.LPF_FilterWrapper(
        baselines.TwoSeg1D("euler_1d", ollson=False), _LPF_CUTOFF_FREQ, None, quiet=True
    ),
    ring.ml.base.LPF_FilterWrapper(
        baselines.TwoSeg1D("1d_corr", ollson=True), _LPF_CUTOFF_FREQ, None, quiet=True
    ),
    ring.ml.base.LPF_FilterWrapper(
        baselines.TwoSeg1D("euler_1d", ollson=True), _LPF_CUTOFF_FREQ, None, quiet=True
    ),
    ring.ml.base.LPF_FilterWrapper(
        baselines.VQF_9D("VQF_9D"), _LPF_CUTOFF_FREQ, 100, quiet=True
    ),
    ring.RING([-1, 0], 0.01),
]
method_names = [
    "1d_corr_ollson_0",
    "euler_1d_ollson_0",
    "1d_corr_ollson_1",
    "euler_1d_ollson_1",
    "VQF_9D",
    "RING",
]


def eval_section_5_3_1B(method, method_name) -> list[float]:
    mae = []
    for i in range(2, 5):
        imtp = IMTP(
            [f"seg{i}", f"seg{i + 1}"],
            joint_axes=True,
            joint_axes_field=True,
            mag=True if method_name == "VQF_9D" else False,
            flex=True,
            dt=False if method_name == "RING" else True,
        )
        for trial in [1, 2]:
            errors, *_ = benchmark(imtp, trial, method, warmup=5.0)
            mae.append(errors[f"seg{i + 1}"]["mae"])
    return mae


if __name__ == "__main__":
    for method, method_name in zip(methods, method_names):
        mae = eval_section_5_3_1B(method, method_name)
        print(f"Method `{method_name}` achieved {np.mean(mae)} +/- {np.std(mae)}")
