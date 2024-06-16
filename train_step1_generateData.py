from pathlib import Path

import fire

import ring
from ring.utils import to_list

prob_rigid: float = 0.25
pos_min_max: float = 0.05
all_rigid_or_flex: bool = True
rand_sampling_rates: bool = True


def main(
    config: str,
    output_path: str,
    size: int,
    seed: int = 1,
    sampling_rates: list[float] = [40, 60, 80, 100, 120, 140, 160, 180, 200],
    anchors: list[str] = [
        "seg3_2Seg",
        "seg4_2Seg",
        "seg3_3Seg",
        "seg5_3Seg",
        "seg2_4Seg",
        "seg3_4Seg",
        "seg4_4Seg",
        "seg5_4Seg",
    ],
):
    sampling_rates = to_list(sampling_rates)
    anchors = to_list(anchors)

    folder = Path(output_path)
    folder.mkdir(exist_ok=True, parents=True)

    size = int(size / len(sampling_rates))

    for i, sampling_rate in enumerate(sampling_rates):
        filepath = folder.joinpath(
            f"data_{config}_{sampling_rate}Hz_size{size}_seed{seed + i}"
        )
        ring.RCMG(
            ring.io.load_example("exclude/standard_sys_rr_imp"),
            ring.MotionConfig.from_register(config),
            add_X_imus=True,
            add_X_jointaxes=True,
            add_X_jointaxes_kwargs=dict(randomly_flip=True),
            add_y_relpose=True,
            add_y_rootincl=True,
            dynamic_simulation=True,
            imu_motion_artifacts=True,
            imu_motion_artifacts_kwargs=dict(
                prob_rigid=prob_rigid,
                pos_min_max=pos_min_max,
                all_imus_either_rigid_or_flex=all_rigid_or_flex,
                imus_surely_rigid=["imu3_1Seg"],
                disable_warning=True,
            ),
            randomize_joint_params=True,
            randomize_motion_artifacts=True,
            randomize_positions=True,
            randomize_anchors=True,
            randomize_anchors_kwargs=dict(anchors=anchors),
            randomize_hz=rand_sampling_rates,
            randomize_hz_kwargs=dict(sampling_rates=[sampling_rate]),
        ).to_pickle(filepath, size, seed=seed + i)
        print(f"Created data file at {str(filepath)}.pickle")


if __name__ == "__main__":
    fire.Fire(main)
