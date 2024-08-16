import time

import jax
import numpy as np
import ring


def to_device(tree, backend: str):
    return jax.tree.map(lambda arr: jax.device_put(arr, jax.devices(backend)[0]), tree)


def time_ring(
    N: int,
    iters: int = 10_000,
    backend=None,
    include_data_transfer_overhead: bool = False,
):
    lam = list(range(-1, N - 1))
    ringnet = ring.RING(lam, Ts=0.01).unwrapped.unwrapped
    # parameters are always already on the correct device
    ringnet.unwrapped.params = to_device(ringnet.search_attr("params"), backend)

    # the sensor input data comes in as numpy arrays
    # (Batchsize, Timesteps, N bodies, 9)
    X = np.zeros((1, 1, N, 9))
    if not include_data_transfer_overhead:
        X = to_device(X, backend)
    # the NN state is the last NN state so it is already on device (cpu or gpu)
    # so we already put it on device to avoid overhead
    _, state = to_device(ringnet.init(1, X, lam, seed=1), backend)

    jit_apply = jax.jit(ringnet.apply, backend=backend)

    def ring_one_timestep(X, state):
        if include_data_transfer_overhead:
            X = to_device(X, backend)
        yhat, state = jit_apply(X=X, state=state)
        yhat.block_until_ready()
        if include_data_transfer_overhead:
            yhat = jax.device_put(yhat, jax.devices("cpu")[0])
        return yhat

    # warmup
    ring_one_timestep(X, state)

    times = [time.perf_counter_ns()]
    for _ in range(iters):
        ring_one_timestep(X, state)
        times.append(time.perf_counter_ns())

    diffs = np.diff(times) / 1000  # from nano to micro seconds
    print(
        f"For N={N}, Overhead={int(include_data_transfer_overhead)}, backend={backend} "
        "one timestep of "
        f"RING takes {np.mean(diffs):.2f} +/- {np.std(diffs):.2f} microseconds "
        f"(evaluated using {iters} function calls)"
    )


if __name__ == "__main__":
    for N in range(1, 5):
        for incl in [False, True]:
            for backend in ["cpu", "gpu", "tpu"]:
                try:
                    time_ring(N, backend=backend, include_data_transfer_overhead=incl)
                except RuntimeError:
                    pass
