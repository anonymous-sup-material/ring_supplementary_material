import baselines
from data import benchmark
from data import IMTP
import numpy as np
import tqdm

import ring


def attitude(method: str):
    return ring.ml.base.GroundTruthHeading_FilterWrapper(baselines.Attitude(method))


methods = [attitude(m) for m in ["vqf", "seel", "mahony", "madgwick", "riann"]]
ringnet = ring.RING([-1], 0.01)
methods += [ringnet]
method_names = [m.unwrapped.name for m in methods[:-1]] + ["RING"]
maes = {name: [] for name in method_names}

for i in tqdm.tqdm(range(1, 6)):
    for method, method_name in tqdm.tqdm(
        zip(methods, method_names), leave=False, total=len(methods)
    ):
        imtp = IMTP([f"seg{i}"], dt=False if method_name == "RING" else True)
        for trial in [1, 2]:
            errors, *_ = benchmark(imtp, trial, method, warmup=5.0)
            maes[method_name].append(errors[f"seg{i}"]["mae"])

for name in method_names:
    mae = maes[name]
    print(f"Method `{name}` achieved {np.mean(mae)} +/- {np.std(mae)}")
