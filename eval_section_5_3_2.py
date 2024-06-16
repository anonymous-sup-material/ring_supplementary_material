from data import benchmark
from data import IMTP
import numpy as np
import tqdm

import ring

# TODO: Investigate why the 100Hz-RING is outperformed here by the sampling rate
# adaptive version
ringnet = ring.RING([-1, 0, 1], None)
method_names = ["RING"]
maes = {name: [] for name in method_names}

for i in tqdm.tqdm(range(2, 4)):
    for method_name in tqdm.tqdm(method_names, leave=False, total=len(method_names)):
        imtp = IMTP(
            [f"seg{i}", f"seg{i + 1}", f"seg{i + 2}"],
            joint_axes=False,
            joint_axes_field=True,
            dt=True,
            sparse=True,
        )

        for trial in [1, 2]:
            errors, *_ = benchmark(imtp, trial, ringnet, warmup=5.0)
            maes[method_name].extend(
                [errors[f"seg{i + 1}"]["mae"], errors[f"seg{i + 2}"]["mae"]]
            )

for name in method_names:
    mae = maes[name]
    print(f"Method `{name}` achieved {np.mean(mae)} +/- {np.std(mae)}")
