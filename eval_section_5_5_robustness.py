import importlib

import numpy as np
import ring
from ring.ml.base import AbstractFilter
import tqdm


class RandomBiasNoiseWrapper(ring.ml.base.AbstractFilterWrapper):
    def __init__(
        self,
        filter: AbstractFilter,
        acc_noise_density: float,  # micro g / sqrt(hz)
        acc_offset: float,  # milli g
        gyr_noise_density: float,  # deg / s / sqrt(hz)
        gyr_offset: float,  # deg / s
        bias=False,
        noise=False,
        bandwidth: float = 200,
        name=None,
    ) -> None:
        super().__init__(filter, name)

        self.acc_noise_density = acc_noise_density
        self.acc_offset = acc_offset
        self.gyr_noise_density = gyr_noise_density
        self.gyr_offset = gyr_offset
        self.bandwidth = bandwidth
        self.bias, self.noise = bias, noise

    def _noise_density_to_std(self, density: float) -> float:
        return density * np.sqrt(self.bandwidth)

    def _get_bias(self, val: float):
        if not self.bias:
            return 0.0
        bias = np.random.uniform(-1.0, 1.0, (3,))
        return bias * val / np.linalg.norm(bias)

    def _get_noise(self, density: float, T: int):
        if not self.noise:
            return 0.0
        return np.random.normal(0.0, self._noise_density_to_std(density), size=(T, 3))

    def apply(self, X: np.ndarray, params=None, state=None, y=None, lam=None):
        T, N, _ = X.shape
        for i in range(N):
            # if both acc and gyr data is zeros, then the body has no IMU, i.e. the
            # IMTP is a sparse IMTP with one or two inner-body IMUs missing; do not
            # add noise and bias in this case
            if np.allclose(X[:, i, :6], 0.0):
                continue

            X[:, i, :3] += self._get_bias(self.acc_offset * 9.81 / 1000)
            X[:, i, 3:6] += self._get_bias(np.deg2rad(self.gyr_offset))
            X[:, i, :3] += self._get_noise(self.acc_noise_density * 9.81 / 1_000_000, T)
            X[:, i, 3:6] += self._get_noise(np.deg2rad(self.gyr_noise_density), T)

        return super().apply(X, params, state, y, lam)


if __name__ == "__main__":

    n_seeds = 10
    n_levels = 7

    factor = 1.2
    max_acc_noise_density = 200 * factor
    max_acc_offset = 40 * factor
    max_gyr_noise_density = 0.03 * factor
    max_gyr_offset = 3 * factor

    data = {}
    for section in tqdm.tqdm(
        ["5_2_1", "5_2_2", "5_2_3", "5_2_4", "5_3_1A", "5_3_1B", "5_3_2", "5_3_3"]
    ):
        module = importlib.import_module(f"eval_section_{section}")
        methods = getattr(module, "methods")
        method_names = getattr(module, "method_names")
        eval_fn = getattr(module, f"eval_section_{section}")

        data[section] = {}
        for method, method_name in tqdm.tqdm(
            zip(methods, method_names), total=len(methods), leave=False
        ):
            data[section][method_name] = {}
            for i in tqdm.tqdm(range(n_levels), leave=False):
                alpha = i / (n_levels - 1)
                data[section][method_name][i] = []
                for j in tqdm.tqdm(range(n_seeds if i != 0 else 1), leave=False):
                    np.random.seed(j)

                    mae = eval_fn(
                        RandomBiasNoiseWrapper(
                            method,
                            acc_noise_density=alpha * max_acc_noise_density,
                            acc_offset=alpha * max_acc_offset,
                            gyr_noise_density=alpha * max_gyr_noise_density,
                            gyr_offset=alpha * max_gyr_offset,
                            noise=True,
                            bias=True,
                            bandwidth=200,  # Hz
                        ),
                        method_name,
                    )

                    data[section][method_name][i].extend(mae)

    ring.utils.pickle_save(data, "eval_section_5_5_robustness.pickle", overwrite=True)
