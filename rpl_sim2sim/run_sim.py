"""Run FAR-FALCON-Terrains' sim2sim environment (vendored) for the RPL depth student.

Usage::

    python -m rpl_sim2sim.run_sim                     # FALCON course, Xbox pad on
    python -m rpl_sim2sim.run_sim --no-joystick       # drive the policy from its terminal
    python -m rpl_sim2sim.run_sim --spawn 0 -2 0.8    # start on the floor facing the stairs

Equivalent to FALCON's ``sim2real/run_sim.py task=loco_manip/loco_manip_cam robot=g1/g1_29dof
+obs=loco_manip/g1_29dof_depth task.USE_JOYSTICK=true task.sim_env.config.ROBOT_SCENE=...
scene_g1_29dof_terrains_freebase_real.xml``. Pair it with ``python -m rpl_sim2sim.run_policy``
(DDS domain 0 on ``lo``, depth in ``depth_img_shm``). MuJoCo window: 7/8 raise/lower the band,
9 releases it, 3-6 move its anchor, R toggles video recording.
"""

from __future__ import annotations

import os

# Offscreen (camera / video) renderers on EGL, the passive viewer on GLFW, as FALCON's run_sim.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # type: ignore[import-untyped]  # noqa: E402

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "config" / "loco_manip_cam.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    scene = Path(config["ROBOT_SCENE"])
    config["ROBOT_SCENE"] = str(scene if scene.is_absolute() else PACKAGE_DIR / scene)
    return config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--no-joystick", action="store_true", help="no Xbox pad (USE_JOYSTICK: false)")
    ap.add_argument("--domain-id", type=int, default=None, help="DDS domain (default: the config's, 0)")
    ap.add_argument("--shm-name", default="depth_img_shm", help="depth shared-memory block")
    ap.add_argument("--spawn", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="pelvis start position; the band anchor moves above it (FALCON: 2.5 -2 1.893)")  # fmt: skip
    ap.add_argument("--video", default="logs/sim2sim/simulation.mp4", help="where R-toggled video is written")
    ap.add_argument("--height-scan", action="store_true",
                    help="send the terrain teachers' height map over UDP (rpl_sim2sim/height_scan.py)")  # fmt: skip
    args = ap.parse_args()

    config = load_config(args.config)
    if args.no_joystick:
        config["USE_JOYSTICK"] = False
    if args.domain_id is not None:
        config["DOMAIN_ID"] = args.domain_id
    config["IMG_SHM_NAME"] = args.shm_name
    Path(args.video).parent.mkdir(parents=True, exist_ok=True)

    import mujoco

    from rpl_sim2sim.simulator import FalconLocoManipCamSimulator

    simulator = FalconLocoManipCamSimulator(config, video_path=args.video)
    simulator.height_scan_enabled = args.height_scan
    if args.spawn is not None:
        simulator.spawn = args.spawn  # SIGUSR1 reset returns here
        simulator.mj_data.qpos[:3] = args.spawn
        mujoco.mj_forward(simulator.mj_model, simulator.mj_data)
        # The default anchor z=3.5 is tuned for the default start on the 0.75 m platform (standing base z 0.793):
        # the 40 kg robot sags ~1.97 m and hangs with its ankles ~4.5 cm above the platform. Keep that offset
        # from the spawn height; leaving z at 3.5 for a floor spawn (Z=0.8) hangs the ankles 0.77 m up.
        simulator.elastic_band.point = np.array([args.spawn[0], args.spawn[1], args.spawn[2] + (3.5 - 0.75 - 0.793)])

    simulator.sim_thread.start()
    if simulator.record_thread:
        simulator.record_thread.start()
    simulator.cam_thread.start()
    simulator.sim_thread.join()


if __name__ == "__main__":
    main()
