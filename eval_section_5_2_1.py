import numpy as np
import ring

import baselines
from data import benchmark
from data import IMTP


def attitude(method: str):
    return ring.ml.base.GroundTruthHeading_FilterWrapper(baselines.Attitude(method))


def eval_section_5_2_1(method, method_name) -> list[float]:
    maes = []
    for i in range(1, 6):
        imtp = IMTP([f"seg{i}"], dt=False if method_name == "RING" else True)
        for trial in [1, 2]:
            errors, *_ = benchmark(imtp, trial, method, warmup=5.0)
            maes.append(errors[f"seg{i}"]["mae"])
    return maes


methods = [attitude(m) for m in ["vqf", "seel", "mahony", "madgwick", "riann"]]
ringnet = ring.RING([-1], 0.01)
methods += [ringnet]
method_names = [m.unwrapped.name for m in methods[:-1]] + ["RING"]


if __name__ == "__main__":

    for method, method_name in zip(methods, method_names):
        mae = eval_section_5_2_1(method)
        print(f"Method `{method_name}` achieved {np.mean(mae)} +/- {np.std(mae)}")
