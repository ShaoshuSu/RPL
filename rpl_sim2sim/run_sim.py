"""Run the FAR-FALCON-Terrains sim2sim environment in Holosoma's MuJoCo ``run_sim``.

Usage::

    python -m rpl_sim2sim.run_sim            # FALCON course, 200 Hz, dual depth -> shm
    python -m rpl_sim2sim.run_sim --robot.init-state.pos 0 0 0.793   # start on the floor

Starts the bridge (Unitree DDS on ``lo``), the virtual gantry, and the ``depth_img_shm``
shared-memory block (``[2, 1, 27, 48]``, front then back) the policy reads. Any
``holosoma.run_sim`` argument can be added or overridden.
"""

from __future__ import annotations

import os
import sys

# Offscreen (depth) rendering on EGL, as FAR-FALCON-Terrains' run_sim does, before anything imports
# mujoco. Without it the depth thread falls back to GLFW: ~25x slower per frame (0.2 ms -> 4-5 ms,
# GIL held), so frames go stale (0.25 s+ instead of 0.1 s) and it races the GLFW viewer's X11 error
# handler (segfaults). The passive viewer itself stays on GLFW.
os.environ.setdefault("MUJOCO_GL", "egl")

from rpl_sim2sim.sim_presets import FALCON_DEPTH_CAMERAS  # noqa: E402  (importing registers the presets)

DEFAULT_ARGS = [
    "simulator:mujoco-falcon-sim2sim",
    "robot:g1-29dof-falcon-sim2sim",
    "terrain:terrain-falcon-sim2sim",
    f"sensor.{FALCON_DEPTH_CAMERAS[0]}:falcon-front-depth",
    f"sensor.{FALCON_DEPTH_CAMERAS[1]}:falcon-back-depth",
    "plugin.depth:depth-shm-falcon",
]


def main() -> None:
    from holosoma.run_sim import main as holosoma_run_sim

    # Defaults first so explicit CLI arguments win.
    given = sys.argv[1:]
    prefixes = {a.split(":", 1)[0] for a in given if ":" in a and not a.startswith("-")}
    defaults = [a for a in DEFAULT_ARGS if a.split(":", 1)[0] not in prefixes]
    sys.argv = [sys.argv[0], *defaults, *given]
    holosoma_run_sim()


if __name__ == "__main__":
    main()
