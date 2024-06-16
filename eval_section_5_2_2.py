import baselines
from data import benchmark
from data import IMTP
import numpy as np
import tqdm

import ring

methods = [
    baselines.TwoSeg1D("1d_corr", ollson=False),
    baselines.TwoSeg1D("euler_1d", ollson=False),
    baselines.VQF_9D("VQF_9D"),
    ring.RING([-1, 0], 0.01),
]
method_names = [m.name for m in methods[:-1]] + ["RING"]
maes = {name: [] for name in method_names}

for i in tqdm.tqdm(range(2, 5)):
    for method, method_name in tqdm.tqdm(
        zip(methods, method_names), leave=False, total=len(methods)
    ):
        imtp = IMTP(
            [f"seg{i}", f"seg{i + 1}"],
            joint_axes=True,
            joint_axes_field=True,
            mag=True if method_name == "VQF_9D" else False,
            dt=False if method_name == "RING" else True,
        )
        for trial in [1, 2]:
            errors, *_ = benchmark(imtp, trial, method, warmup=5.0)
            maes[method_name].append(errors[f"seg{i + 1}"]["mae"])

for name in method_names:
    mae = maes[name]
    print(f"Method `{name}` achieved {np.mean(mae)} +/- {np.std(mae)}")
