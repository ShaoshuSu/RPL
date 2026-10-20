"""FALCON's G1 arm IK for joystick end-effector targets (``sim2real.utils.arm_ik.ArmIK_NoRotation``).

Solves the shoulder + elbow joints of both arms so a point 0.35 m down each forearm (FALCON's
``L_ee`` / ``R_ee``) reaches a pelvis-frame target, with legs, waist and wrists locked at zero.
Translation only (FALCON's G1 config uses the no-rotation solver), ipopt via casadi, targets
low-passed toward the command and solutions smoothed with FALCON's weighted moving filter.
Collision checking, meshcat and external-force torques are dropped: FALCON never enables them
in ``get_q_tau``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import casadi
import numpy as np
import pinocchio as pin
from loguru import logger
from pinocchio import casadi as cpin

# FALCON's ``humanoidverse/data/robots/g1/g1_29dof.urdf`` (same kinematics as the sim's MJCF).
URDF_PATH = Path(__file__).resolve().parent / "assets" / "g1_29dof.urdf"

# FALCON ``g1_29dof.yaml`` ``IK_SOLVER.robot_cfg``.
LOCKED_JOINTS = [
    *(
        f"{side}_{j}_joint"
        for side in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ),
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
FIXED_UPPER_JOINTS = [f"{side}_wrist_{j}_joint" for side in ("left", "right") for j in ("roll", "pitch", "yaw")]
L_EE = ("left_elbow_joint", (0.35, -0.075, 0.0))
R_EE = ("right_elbow_joint", (0.35, 0.075, 0.0))


class WeightedMovingFilter:
    """FALCON's ``WeightedMovingFilter``: weighted average of the last ``len(weights)`` distinct samples."""

    def __init__(self, weights, data_size: int):
        self._weights = np.asarray(weights)
        assert np.isclose(self._weights.sum(), 1.0)
        self._queue: list[np.ndarray] = []
        self.filtered_data = np.zeros(data_size)

    def add_data(self, new_data: np.ndarray) -> None:
        if self._queue and np.array_equal(new_data, self._queue[-1]):
            return
        if len(self._queue) >= len(self._weights):
            self._queue.pop(0)
        self._queue.append(new_data)
        if len(self._queue) < len(self._weights):
            self.filtered_data = new_data
        else:
            # np.convolve(..., "valid")[-1] in FALCON: the newest sample gets weights[0].
            self.filtered_data = self._weights @ np.asarray(self._queue)[::-1]


class ArmIK:
    """Two-arm position IK on the reduced G1 model (shoulders + elbows)."""

    def __init__(self, upper_dof_names: list[str], urdf_path: Path = URDF_PATH):
        full = pin.buildModelFromUrdf(str(urdf_path))  # kinematics only: no meshes needed
        model = pin.buildReducedModel(
            full, [full.getJointId(n) for n in LOCKED_JOINTS + FIXED_UPPER_JOINTS], np.zeros(full.nq)
        )
        for name, (joint, offset) in (("L_ee", L_EE), ("R_ee", R_EE)):
            model.addFrame(
                pin.Frame(name, model.getJointId(joint), pin.SE3(np.eye(3), np.array(offset)), pin.FrameType.OP_FRAME)
            )
        self.model = model
        # Upper-body indices the solution writes (FALCON's ``active_joint_indices``), in model order.
        self.active_joint_indices = [upper_dof_names.index(model.names[j]) for j in range(1, model.njoints)]

        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        cq = casadi.SX.sym("q", model.nq, 1)
        cp_l = casadi.SX.sym("p_l", 3, 1)
        cp_r = casadi.SX.sym("p_r", 3, 1)
        cpin.framesForwardKinematics(cmodel, cdata, cq)
        l_id, r_id = model.getFrameId("L_ee"), model.getFrameId("R_ee")
        translational_error = casadi.Function(
            "translational_error",
            [cq, cp_l, cp_r],
            [casadi.vertcat(cdata.oMf[l_id].translation - cp_l, cdata.oMf[r_id].translation - cp_r)],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(model.nq)
        self.var_q_last = self.opti.parameter(model.nq)
        self.param_p_l = self.opti.parameter(3)
        self.param_p_r = self.opti.parameter(3)
        self.opti.subject_to(self.opti.bounded(model.lowerPositionLimit, self.var_q, model.upperPositionLimit))
        self.opti.minimize(
            50 * casadi.sumsqr(translational_error(self.var_q, self.param_p_l, self.param_p_r))
            + 0.02 * casadi.sumsqr(self.var_q)
            + 0.1 * casadi.sumsqr(self.var_q - self.var_q_last)
        )
        self.opti.solver(
            "ipopt",
            {"ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-6}, "print_time": False, "calc_lam_p": False},
        )

        self.speed_factor = 0.02
        self.current_p_l: np.ndarray | None = None
        self.current_p_r: np.ndarray | None = None
        self.init_data = np.zeros(model.nq)
        self.smooth_filter = WeightedMovingFilter([0.4, 0.3, 0.2, 0.1], model.nq)

    def set_initial_targets(self, p_l: np.ndarray, p_r: np.ndarray) -> None:
        self.current_p_l = np.array(p_l, dtype=np.float64)
        self.current_p_r = np.array(p_r, dtype=np.float64)

    def get_q(self, p_l_target: np.ndarray, p_r_target: np.ndarray, current_upper_q: np.ndarray) -> np.ndarray:
        """Step the interpolated targets toward the command and solve (FALCON's ``get_q_tau``).

        ``current_upper_q`` is the measured upper-body joint vector (``upper_dof_names`` order);
        returns the active-joint solution, or the measured active joints if ipopt fails.
        """
        s = self.speed_factor
        self.current_p_l = (1 - s) * self.current_p_l + s * np.asarray(p_l_target)
        self.current_p_r = (1 - s) * self.current_p_r + s * np.asarray(p_r_target)
        current_q = np.asarray(current_upper_q, dtype=np.float64)[self.active_joint_indices]

        self.init_data = current_q
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_p_l, self.current_p_l)
        self.opti.set_value(self.param_p_r, self.current_p_r)
        self.opti.set_value(self.var_q_last, self.init_data)
        try:
            self.opti.solve()
            self.smooth_filter.add_data(self.opti.value(self.var_q))
            self.init_data = self.smooth_filter.filtered_data
            return self.init_data
        except RuntimeError as e:
            logger.warning(f"[ArmIK] no convergence ({e}); holding the measured arm pose")
            self.smooth_filter.add_data(self.opti.debug.value(self.var_q))
            self.init_data = self.smooth_filter.filtered_data
            return current_q


class ArmIKThread:
    """Runs :class:`ArmIK` off the control loop at a fixed rate (FALCON's ``upper_body_worker_thread``).

    The control loop calls :meth:`submit` with the measured upper-body joints and the EE targets
    every tick and :meth:`latest` for the newest solution; the solver never blocks it.
    """

    def __init__(self, ik: ArmIK, ref_upper_dof_pos: np.ndarray, frequency_hz: float = 30.0):
        self.ik = ik
        self.period = 1.0 / frequency_hz
        self._lock = threading.Lock()
        self._input: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._output = np.array(ref_upper_dof_pos, dtype=np.float32).reshape(1, -1)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ArmIK")
        self._thread.start()
        logger.info(f"[ArmIK] upper-body IK thread started at {1.0 / self.period:.0f} Hz")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def submit(self, current_upper_q: np.ndarray, p_l: np.ndarray, p_r: np.ndarray) -> None:
        with self._lock:
            self._input = (np.array(current_upper_q), np.array(p_l), np.array(p_r))

    def latest(self) -> np.ndarray:
        with self._lock:
            return self._output.copy()

    def _run(self) -> None:
        last = time.perf_counter()
        while self._running:
            sleep = self.period - (time.perf_counter() - last)
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -0.1:
                logger.warning(f"[ArmIK] running {-sleep * 1000:.0f} ms behind schedule")
            last = time.perf_counter()
            with self._lock:
                job, self._input = self._input, None
            if job is None:
                continue
            q_upper, p_l, p_r = job
            try:
                sol = self.ik.get_q(p_l, p_r, q_upper)
            except Exception as e:  # keep the thread alive; the last solution stays in place
                logger.error(f"[ArmIK] solver error: {e}")
                continue
            with self._lock:
                self._output[0, self.ik.active_joint_indices] = sol
