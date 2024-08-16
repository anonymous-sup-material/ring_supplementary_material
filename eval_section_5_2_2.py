import numpy as np
import ring

import baselines
from data import benchmark
from data import IMTP


def eval_section_5_2_2(method, method_name) -> list[float]:
    mae = []
    for i in range(2, 5):
        imtp = IMTP(
            [f"seg{i}", f"seg{i + 1}"],
            joint_axes=True,
            joint_axes_field=True,
            mag=True if method_name == "VQF_9D" else False,
            dt=False if method_name == "RING" else True,
        )
        for trial in [1, 2]:
            errors, *_ = benchmark(imtp, trial, method, warmup=5.0)
            mae.append(errors[f"seg{i + 1}"]["mae"])
    return mae


methods = [
    baselines.TwoSeg1D("1d_corr", ollson=False),
    baselines.TwoSeg1D("euler_1d", ollson=False),
    baselines.VQF_9D("VQF_9D"),
    ring.RING([-1, 0], 0.01),
]
method_names = [m.name for m in methods[:-1]] + ["RING"]


if __name__ == "__main__":

    for method, method_name in zip(methods, method_names):
        mae = eval_section_5_2_2(method, method_name)
        print(f"Method `{method_name}` achieved {np.mean(mae)} +/- {np.std(mae)}")
