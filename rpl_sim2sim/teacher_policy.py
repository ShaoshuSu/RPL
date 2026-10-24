"""RPL terrain teacher (privileged height-map expert) for ``holosoma_inference``.

Same observation pipeline as the depth student (RplDepthPolicy: 115-dim state frame x 5, time-major), with the
16 x 10 height map from the sim (height_scan.py, run_sim.py --height-scan) in place of the depth images:
``actor_obs_lower_body = [actor_obs | perception_obs]``. Models come from tools/export_teacher_onnx.py; each
teacher was trained on one terrain type (stairs_up, stairs_down, slopes, stepping_stones_columns).

    python -m rpl_sim2sim.run_policy inference:g1-29dof-rpl-teacher \
        --task.model-path rpl_sim2sim/models/rpl_teacher_stairs_down_model_74998.onnx
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from rpl_sim2sim.height_scan import HeightScanReceiver
from rpl_sim2sim.policy import RplDepthPolicy

# Like the student's depth_shm stale_warn_s / stale_stop_s: warn, then stop the policy (init pose).
SCAN_STALE_WARN_S = 0.1
SCAN_STALE_STOP_S = 0.5


class RplHeightmapTeacherPolicy(RplDepthPolicy):
    LOWER_BODY_GROUPS = ("actor_obs", "perception_obs")

    def _init_depth_sensor(self) -> None:
        self.height_scan = HeightScanReceiver()
        self.height_scan.start()
        self._scan_stale_warned = False
        logger.info("[RplHeightmapTeacherPolicy] waiting for height maps from run_sim.py --height-scan")

    def _perception_obs(self) -> dict[str, np.ndarray]:
        return {"height_scan": self.height_scan.latest().reshape(1, -1)}

    def _check_depth_liveness(self) -> None:
        age = self.height_scan.age_s()
        if age is None:
            if self.use_policy_action:
                logger.error("[RplHeightmapTeacherPolicy] no height map yet (run_sim.py --height-scan?): stopping policy")
                self._handle_init_state()
            return
        if age > SCAN_STALE_WARN_S and not self._scan_stale_warned:
            logger.warning(f"[RplHeightmapTeacherPolicy] height map {age:.2f}s old")
            self._scan_stale_warned = True
        elif age <= SCAN_STALE_WARN_S and self._scan_stale_warned:
            logger.info("[RplHeightmapTeacherPolicy] height maps arriving again")
            self._scan_stale_warned = False
        if age > SCAN_STALE_STOP_S and self.use_policy_action:
            logger.error(f"[RplHeightmapTeacherPolicy] height map {age:.2f}s old: stopping policy")
            self._handle_init_state()
