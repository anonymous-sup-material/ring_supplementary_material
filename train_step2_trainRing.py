import os
from pathlib import Path
import random

import fire
import jax.numpy as jnp
import numpy as np
import ring
from ring import maths
from ring import ml
from ring.algorithms.generator import transforms
import wandb

dropout_rates = dict(
    seg3_1Seg=(0.0, 1.0),
    seg3_2Seg=(0.0, 1.0),
    seg4_2Seg=(0.0, 0.5),
    seg3_3Seg=(0.0, 1.0),
    seg4_3Seg=(2 / 3, 0.5),
    seg5_3Seg=(0.0, 0.5),
    seg2_4Seg=(0.0, 1.0),
    seg3_4Seg=(3 / 4, 1 / 4),
    seg4_4Seg=(3 / 4, 1 / 4),
    seg5_4Seg=(0.0, 1 / 4),
)


def output_transform_factory(link_names):

    def _rename_links(d: dict[str, dict]):
        for key in list(d.keys()):
            if key in link_names:
                d[str(link_names.index(key))] = d.pop(key)

    def output_transform(tree):
        X, y = tree
        segments = list(set(X.keys()) - set(["dt"]))

        any_segment = X[segments[0]]
        assert any_segment["gyr"].ndim == 3, f"{any_segment['gyr'].shape}"
        B = any_segment["gyr"].shape[0]

        draw = lambda p: np.random.binomial(1, p, size=B).astype(float)[:, None, None]
        fcs = {
            "seg4_3Seg": draw(1 - dropout_rates["seg4_3Seg"][1]),
            "seg5_3Seg": draw(1 - dropout_rates["seg5_3Seg"][1]),
        }

        for segments, (imu_rate, jointaxes_rate) in dropout_rates.items():
            factor_imu = draw(1 - imu_rate)
            factor_ja = draw(1 - jointaxes_rate)

            if segments == "seg4_3Seg":
                factor_imu = 0.0

            if segments in fcs:
                factor_ja = fcs[segments]

            for gyraccmag in ["gyr", "acc", "mag"]:
                if gyraccmag in X[segments]:
                    X[segments][gyraccmag] *= factor_imu

            if "joint_axes" in X[segments]:
                X[segments]["joint_axes"] *= factor_ja

        _rename_links(X)
        _rename_links(y)
        return transforms._expand_then_flatten((X, y))

    return output_transform


def _make_ring(lam, params_warmstart: str | None, dry_run: bool):
    hidden_state_dim = 400 if not dry_run else 20
    message_dim = 200 if not dry_run else 10
    ringnet = ml.RING(
        lam=lam,
        hidden_state_dim=hidden_state_dim,
        message_dim=message_dim,
        params=params_warmstart,
    )
    ringnet = ml.base.ScaleX_FilterWrapper(ringnet)
    ringnet = ml.base.GroundTruthHeading_FilterWrapper(ringnet)
    return ringnet


def main(
    bs: int,
    episodes: int,
    path_data_folder: str,
    path_trained_params: str,
    use_wandb: bool = False,
    wandb_project: str = "RING",
    params_warmstart: str = None,
    seed: int = 1,
    dry_run: bool = False,
):

    random.seed(seed)
    np.random.seed(seed)

    if use_wandb:
        wandb.init(project=wandb_project)

    sys_noimu = ring.io.load_example("exclude/standard_sys_rr_imp").make_sys_noimu()[0]

    ringnet = _make_ring(sys_noimu.link_parents, params_warmstart, dry_run)

    generator, (X_val, y_val) = ml.ml_utils.train_val_split(
        [
            Path(path_data_folder).joinpath(file)
            for file in os.listdir(path_data_folder)
        ],
        bs,
        transform_gen=transforms.GeneratorTrafoLambda(
            output_transform_factory(sys_noimu.link_names)
        ),
    )

    _mae_metrices = dict(
        mae_deg=lambda q, qhat: jnp.rad2deg(
            jnp.mean(maths.angle_error(q, qhat)[:, 2500:])
        )
    )

    callbacks = [
        ml.callbacks.EvalXyTrainingLoopCallback(
            ringnet,
            _mae_metrices,
            X_val,
            y_val,
            None,
            "val",
            link_names=sys_noimu.link_names,
        )
    ]

    optimizer = ml.make_optimizer(
        1e-3,
        episodes,
        n_steps_per_episode=6,
        skip_large_update_max_normsq=100.0,
    )

    ml.train_fn(
        generator,
        episodes,
        ringnet,
        optimizer=optimizer,
        callbacks=callbacks,
        callback_kill_after_seconds=23.5 * 3600,
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        callback_save_params=path_trained_params,
        seed_network=seed,
        link_names=sys_noimu.link_names,
    )


if __name__ == "__main__":
    fire.Fire(main)
