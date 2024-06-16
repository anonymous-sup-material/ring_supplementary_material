import numpy as np
import qmt

from ring.ml import base as ml_base

from .riann import RIANN


def _riann_predict(gyr, acc, params: dict):
    fs = 1 / params["Ts"]
    riann = RIANN()
    return riann.predict(acc, gyr, fs)


_attitude_methods = {
    "vqf": ("VQFAttitude", qmt.oriEstVQF),
    "madgwick": ("MadgwickAttitude", qmt.oriEstMadgwick),
    "mahony": ("MahonyAttitude", qmt.oriEstMahony),
    "seel": ("SeelAttitude", qmt.oriEstIMU),
    "riann": ("RIANN", _riann_predict),
}


class Attitude(ml_base.AbstractFilterUnbatched):
    def __init__(self, method: str) -> None:
        """ """
        self._name, self._method = _attitude_methods[method]

    def _apply_unbatched(self, X, params, state, y, lam):

        T, N, F = X.shape
        assert F == 10 or F == 7
        dt = float(X[0, 0, -1])

        quats = np.zeros((T, N, 4))
        for i in range(N):
            quats[:, i] = self._method(
                gyr=X[:, i, 3:6], acc=X[:, i, :3], params=dict(Ts=dt)
            )

        # NOTE CONVENTION !!
        quats = qmt.qinv(quats)
        return quats, state


class TwoSeg1D(ml_base.AbstractFilterUnbatched):
    def __init__(self, method: str, ollson: bool = False):
        self.method = method
        self.ollson = ollson
        assert method in ["euler_2d", "euler_1d", "1d_corr", "proj"]
        self._name = method + f"_ollson_{int(ollson)}"

    def _apply_unbatched(self, X, params, state, y, lam):

        T, N, F = X.shape
        assert N == 2
        assert tuple(lam) == (-1, 0)
        dt = float(X[0, 0, -1])
        acc1, acc2 = X[:, 0, :3], X[:, 1, :3]
        gyr1, gyr2 = X[:, 0, 3:6], X[:, 1, 3:6]

        if self.ollson:
            axis = qmt.jointAxisEstHingeOlsson(
                acc1,
                acc2,
                gyr1,
                gyr2,
                estSettings=dict(quiet=True),
            )[0][:, 0]
        else:
            assert F == 10
            axis = X[0, 1, 6:9]

        quats = qmt.qinv(Attitude("vqf").apply(X, params, None, y, lam)[0])
        ts = np.arange(T * dt, step=dt)
        quats[:, 1] = qmt.headingCorrection(
            gyr1,
            gyr2,
            quats[:, 0],
            quats[:, 1],
            ts,
            axis,
            None,
            estSettings=dict(constraint=self.method),
        )[0]

        # NOTE CONVENTION !!
        quats = qmt.qinv(quats)
        quats = _absolute_to_relative_orientations(quats, lam)
        return quats, state


def _absolute_to_relative_orientations(quats, lam):
    _quats = quats.copy()
    for i, p in enumerate(lam):
        if p == -1:
            continue
        q1, q2 = quats[:, p], quats[:, i]
        _quats[:, i] = qmt.qmult(q1, qmt.qinv(q2))
    return _quats


class VQF_9D(ml_base.AbstractFilterUnbatched):
    def __init__(
        self,
        name: str,
    ):
        self._name = name

    def _apply_unbatched(self, X, params, state, y, lam):

        assert lam is not None
        T, N, F = X.shape
        assert F == 13 or F == 10
        dt = float(X[0, 0, -1])

        quats = np.zeros((T, N, 4))
        for i in range(N):
            quats[:, i] = qmt.oriEstVQF(
                gyr=X[:, i, 3:6], acc=X[:, i, :3], mag=X[:, i, 6:9], params=dict(Ts=dt)
            )

        # NOTE CONVENTION !!
        quats = qmt.qinv(quats)
        quats = _absolute_to_relative_orientations(quats, lam)
        return quats, state
