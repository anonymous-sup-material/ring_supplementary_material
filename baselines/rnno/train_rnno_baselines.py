import random
from typing import Optional

from diodem.benchmark import benchmark
from diodem.benchmark import IMTP
import fire
import numpy as np

import ring
from ring import ml
from ring.algorithms.generator import transforms
import wandb


def _make_gen(
    sys: ring.System,
    configs: list[str],
    bs: int,
    dry_run: bool,
    eager_gen_size: int | None,
):

    if dry_run:
        randomize_anchors = False
    else:
        randomize_anchors = True

    rcmg = ring.RCMG(
        sys,
        [ring.MotionConfig.from_register(name) for name in configs],
        add_X_imus=True,
        add_y_relpose=True,
        add_y_rootincl=True,
        randomize_anchors=randomize_anchors,
        randomize_positions=True,
        use_link_number_in_Xy=True,
    )
    if eager_gen_size is not None:
        gen = rcmg.to_eager_gen(bs, eager_gen_size)
    else:
        gen = rcmg.to_lazy_gen(bs)
    return transforms.GeneratorTrafoExpandFlatten(gen, jit=True)


def main(
    bs: int,
    episodes: int,
    configs: list[str],
    use_wandb: bool = False,
    wandb_project: str = "universal",
    wandb_name: str = None,
    lr: float = 3e-3,
    seed: int = 1,
    kill_ep: Optional[int] = None,
    kill_after_hours: float = None,
    eager_gen_size: Optional[int] = None,
    xy: bool = False,
    v1: bool = False,
):
    random.seed(seed)
    np.random.seed(seed)

    dry_run = not ml.on_cluster()

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_name, config=locals())

    sys = ring.io.load_example("exclude/standard_sys").delete_system(
        ["seg3_1Seg", "seg3_2Seg", "seg2_4Seg", "imu4_3Seg"]
    )
    if xy:
        sys = sys.change_joint_type("seg4_3Seg", "rx", new_damp=np.array([3.0]))
        sys = sys.change_joint_type("seg5_3Seg", "ry", new_damp=np.array([3.0]))
    sys_noimu = sys.make_sys_noimu()[0]

    if v1:
        ringnet = ring.ml.RNNO(
            12,
            return_quats=True,
            eval=False,
            rnn_layers=(20, 20) if dry_run else (200, 200),
            linear_layers=(20,) if dry_run else (100, 50, 50),
            v1=True,
        )
    else:
        ringnet = ring.ml.RNNO(
            12,
            return_quats=True,
            eval=False,
            hidden_state_dim=20 if dry_run else 200,
        )

    callbacks, metrices_name = [], []

    def add_callback(imtp: IMTP, exp_id, motion_start):
        cb = benchmark(
            imtp=imtp,
            exp_id=exp_id,
            motion_start=motion_start,
            filter=ringnet,
            return_cb=True,
        )
        callbacks.append(cb)
        for segment in imtp.segments:
            metrices_name.append([cb.metric_identifier, "mae_deg", segment])

    # 4SEG exp callbacks
    timings = {
        1: ["slow1", "fast"],
        2: ["slow_fast_mix", "slow_fast_freeze_mix"],
    }
    for exp_id in timings:
        for phase in timings[exp_id]:
            for suffix, chain in zip(
                ["xy", "yz"], [["seg2", "seg3", "seg4"], ["seg3", "seg4", "seg5"]]
            ):
                add_callback(
                    IMTP(
                        chain,
                        joint_axes=False,
                        joint_axes_field=False,
                        sparse=True,
                        model_name_suffix=suffix,
                        dt=False,
                    ),
                    exp_id,
                    phase,
                )

    # create one large "experimental validation" metric
    for zoom_in in metrices_name:
        print(zoom_in)
    callbacks += [ml.callbacks.AverageMetricesTLCB(metrices_name, "exp_val_mae_deg")]

    optimizer = ml.make_optimizer(
        lr,
        episodes,
        n_steps_per_episode=6,
        skip_large_update_max_normsq=100.0,
    )

    ml.train_fn(
        _make_gen(sys, configs, bs, dry_run, eager_gen_size),
        episodes,
        ringnet,
        optimizer=optimizer,
        callbacks=callbacks,
        callback_kill_after_seconds=(
            23.5 * 3600 if kill_after_hours is None else kill_after_hours * 3600
        ),
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        callback_save_params=f"~/params/{ml.unique_id()}.pickle",
        callback_kill_after_episode=kill_ep,
        callback_save_params_track_metrices=[["exp_val_mae_deg"]],
        seed_network=seed,
        link_names=sys_noimu.link_names,
    )


if __name__ == "__main__":
    fire.Fire(main)
