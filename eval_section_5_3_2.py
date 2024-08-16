import numpy as np
import ring

from data import benchmark
from data import IMTP

# TODO: Investigate why the 100Hz-RING is outperformed here by the sampling rate
# adaptive version
methods = [ring.RING([-1, 0, 1], None)]
method_names = ["RING"]


def eval_section_5_3_2(method, method_name) -> list[float]:
    mae = []
    for i in range(2, 4):
        imtp = IMTP(
            [f"seg{i}", f"seg{i + 1}", f"seg{i + 2}"],
            joint_axes=False,
            joint_axes_field=True,
            dt=True,
            sparse=True,
        )

        for trial in [1, 2]:
            errors, *_ = benchmark(imtp, trial, method, warmup=5.0)
            mae.extend([errors[f"seg{i + 1}"]["mae"], errors[f"seg{i + 2}"]["mae"]])

    return mae


if __name__ == "__main__":
    for method, method_name in zip(methods, method_names):
        mae = eval_section_5_3_2(method, method_name)
        print(f"Method `{method_name}` achieved {np.mean(mae)} +/- {np.std(mae)}")
