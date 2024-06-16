import baselines
from data import benchmark
from data import IMTP
import numpy as np
import tqdm

import ring

ringnet = ring.RING([-1, 0, 1], 0.01)
method_names = ["RNNO", "RING"]
maes = {name: [] for name in method_names}

for i in tqdm.tqdm(range(2, 4)):
    for method_name in tqdm.tqdm(method_names, leave=False, total=len(method_names)):
        imtp = IMTP(
            [f"seg{i}", f"seg{i + 1}", f"seg{i + 2}"],
            joint_axes=True if method_name == "RING" else False,
            joint_axes_field=True if method_name == "RING" else False,
            dt=False,
            sparse=True,
        )
        if method_name == "RING":
            method = ringnet
        elif i == 2:
            method = baselines.RNNO_v2_xy

        elif i == 3:
            method = baselines.RNNO_v2_yz
        else:
            raise NotImplementedError

        for trial in [1, 2]:
            errors, *_ = benchmark(imtp, trial, method, warmup=5.0)
            maes[method_name].extend(
                [errors[f"seg{i + 1}"]["mae"], errors[f"seg{i + 2}"]["mae"]]
            )

for name in method_names:
    mae = maes[name]
    print(f"Method `{name}` achieved {np.mean(mae)} +/- {np.std(mae)}")
