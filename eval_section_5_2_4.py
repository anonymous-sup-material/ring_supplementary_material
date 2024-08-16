import numpy as np
import ring
from ring.ml.base import AbstractFilter

import baselines
from data import benchmark
from data import IMTP


def eval_section_5_2_4(method: AbstractFilter, method_name) -> list[float]:
    mae = []
    for i in range(2, 4):
        mode = method.search_attr("mode")
        if i == 2:
            mode("xy")
        else:
            mode("yz")

        imtp = IMTP(
            [f"seg{i}", f"seg{i + 1}", f"seg{i + 2}"],
            joint_axes=True if method_name == "RING" else False,
            joint_axes_field=True if method_name == "RING" else False,
            dt=False,
            sparse=True,
        )
        for trial in [1, 2]:
            errors, *_ = benchmark(imtp, trial, method, warmup=5.0)
            mae.extend([errors[f"seg{i + 1}"]["mae"], errors[f"seg{i + 2}"]["mae"]])
    return mae


class XY_YZ_Wrapper(ring.ml.base.AbstractFilterWrapper):
    def __init__(self, filters: dict[str, AbstractFilter], name=None) -> None:
        super().__init__(None, name)
        self.filters = filters

    def mode(self, mode: str):
        super().__init__(self.filters[mode])


ringnet = ring.RING([-1, 0, 1], 0.01)
methods = [
    XY_YZ_Wrapper({"xy": baselines.RNNO_v2_xy, "yz": baselines.RNNO_v2_yz}),
    XY_YZ_Wrapper(dict(xy=ringnet, yz=ringnet)),
]
method_names = ["RNNO", "RING"]


if __name__ == "__main__":

    for method, method_name in zip(methods, method_names):
        mae = eval_section_5_2_4(method, method_name)
        print(f"Method `{method_name}` achieved {np.mean(mae)} +/- {np.std(mae)}")
