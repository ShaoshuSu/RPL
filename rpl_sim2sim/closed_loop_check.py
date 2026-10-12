"""Headless closed-loop check: the depth student walks FALCON's stair course in MuJoCo.

Runs the FALCON sim2sim environment in this process (headless) and the policy in a child process
driven through a pseudo-terminal (``i`` init pose, ``]`` start, ``=`` walk, ``w`` x N forward
speed), releases the virtual gantry, then logs the pelvis trajectory.

Usage (sim env, e.g. ``rplmujoco``; the policy runs in ``--policy-python``)::

    python -m rpl_sim2sim.closed_loop_check --policy-python ~/.holosoma_deps/miniconda3/envs/rplinference/bin/python

The robot starts on the floor in front of the stairs (x=0, y=-2) facing +x, walks the stairs and the
two-column stepping stones, and is told to stand once it reaches the far platform (``--stop-x``).
Exit status 0 when the pelvis gets onto that platform (``--success-x``) and never drops below
``--fall-z`` above the course surface under it.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pty
import select
import subprocess
import sys
import threading
import time

import numpy as np


def _send(fd: int, keys: str, gap: float = 0.3) -> None:
    for k in keys:
        os.write(fd, k.encode())
        time.sleep(gap)


def _drain(fd: int, log) -> None:
    while True:
        r, _, _ = select.select([fd], [], [], 0.1)
        if not r:
            return
        try:
            data = os.read(fd, 65536)
        except OSError:
            return
        if not data:
            return
        log.write(data.decode(errors="replace"))


def _ground_below(sim, pos) -> float:
    """Height of the course surface (scene geoms, group 2) under ``pos``; 0 if nothing is hit."""
    import mujoco

    start = np.array([pos[0], pos[1], pos[2] + 2.0])
    groups = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(sim.root_model, sim.root_data, start, np.array([0.0, 0.0, -1.0]), groups, 1, -1, geomid)
    return float(start[2] - dist) if dist >= 0 else 0.0


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")  # as rpl_sim2sim.run_sim: EGL depth rendering
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-python", default=sys.executable)
    ap.add_argument("--model-path", default=None, help="ONNX student (default: the shipped one)")
    ap.add_argument("--speed-presses", type=int, default=5, help="'w' presses; 0.1 m/s each")
    ap.add_argument("--walk-seconds", type=float, default=25.0)
    ap.add_argument("--success-x", type=float, default=5.8, help="pelvis x on the far platform (stairs + stones done)")
    ap.add_argument("--stop-x", type=float, default=5.9, help="zero the velocity command once the pelvis passes this x")
    ap.add_argument("--fall-z", type=float, default=0.45)
    ap.add_argument("--log", default="/tmp/rpl_closed_loop_policy.log")
    ap.add_argument("--json", default=None, help="write the pelvis track + verdict here")
    ap.add_argument(
        "--domain-id",
        type=int,
        default=37,
        help="DDS domain for this check's bridge + policy, kept off the default 0 so a concurrently running "
        "interactive run_sim/run_policy never receives this check's motor commands",
    )
    ap.add_argument("--shm-name", default="rpl_closed_loop_depth", help="depth shm block for this check")
    ap.add_argument(
        "--no-heading-hold",
        action="store_true",
        help="send no yaw corrections. By default the check steers like an operator / training's heading "
        "command: q/e presses turn toward the course centreline y=-2",
    )
    ap.add_argument(
        "--training-camera-poses",
        action="store_true",
        help="mount the cameras at the training offsets (x=0.135/-0.125) instead of FALCON's sim ones",
    )
    args, sim_args = ap.parse_known_args()  # anything else goes to run_sim (e.g. --plugin.depth.resize-mode)

    from holosoma.config_types.run_sim import RunSimConfig
    from holosoma.utils.config_registry import parse_config
    from holosoma.utils.sim_utils import DirectSimulation, setup_simulation_environment

    import rpl_sim2sim.sim_presets  # noqa: F401  (registers the presets)
    from rpl_sim2sim.run_sim import DEFAULT_ARGS

    isolation = [
        "--simulator.config.bridge.domain-id",
        str(args.domain_id),
        "--plugin.depth.shm-name",
        args.shm_name,
        "--simulator.config.bridge.use-joystick",
        "False",
    ]
    cfg = parse_config(RunSimConfig, args=[*DEFAULT_ARGS, "--training.headless", "True", *isolation, *sim_args])
    # On the floor in front of the stairs, facing +x (FALCON spawns on the stair platform instead).
    robot = dataclasses.replace(cfg.robot, init_state=dataclasses.replace(cfg.robot.init_state, pos=[0.0, -2.0, 0.8]))
    cfg = dataclasses.replace(cfg, robot=robot)
    if args.training_camera_poses:
        offsets = {"front_depth": [0.135, 0.06, 0.02], "back_depth": [-0.125, -0.06, 0.02]}
        sensors = {
            k: dataclasses.replace(v, mount=dataclasses.replace(v.mount, position=offsets[k]))
            for k, v in cfg.sensor.items()
        }
        cfg = dataclasses.replace(cfg, sensor=sensors)
    env, device, app = setup_simulation_environment(cfg, device="cpu")
    sim_runner = DirectSimulation(cfg, env, device, app)
    sim_runner.initialize()
    sim = sim_runner.simulator
    # Band slack so the robot stands on its feet (anchor height minus standing pelvis height),
    # what pressing "8" in the viewer does by hand.
    import mujoco

    gantry = sim.virtual_gantry
    sag = mujoco.mj_getTotalmass(sim.root_model) * 9.81 / gantry.stiffness
    gantry.length = max(0.0, gantry.height - 0.8 - sag)
    thread = threading.Thread(target=sim_runner.run, daemon=True)
    thread.start()
    time.sleep(2.0)

    cmd = [
        args.policy_python,
        "-m",
        "rpl_sim2sim.run_policy",
        "--task.domain-id",
        str(args.domain_id),
        "--task.depth-shm.name",
        args.shm_name,
        # Driven by key presses on its terminal, not the default joystick.
        "--task.velocity-input",
        "keyboard",
        "--task.state-input",
        "keyboard",
    ]
    if args.model_path:
        cmd += ["--task.model-path", args.model_path]
    master, slave = pty.openpty()
    policy = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    log = open(args.log, "w")
    track: list[tuple[float, ...]] = []  # (t, x, y, z, ground z, yaw)
    ok = False
    try:
        t_end = time.time() + 40.0
        while time.time() < t_end:  # wait for the policy loop
            _drain(master, log)
            log.flush()
            if "Policy initialized successfully" in open(args.log).read() or policy.poll() is not None:
                break
            time.sleep(0.5)
        if policy.poll() is not None:
            print(f"policy exited early ({policy.returncode}); see {args.log}")
            return 2

        def probe(tag: str) -> None:
            p = sim.robot_root_states[0, :3].detach().cpu().numpy()
            q = sim.dof_pos[0].detach().cpu().numpy() if hasattr(sim, "dof_pos") else np.zeros(1)
            quat = sim.robot_root_states[0, 3:7].detach().cpu().numpy().astype(float)
            yaw = np.degrees(
                np.arctan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
            )
            print(f"[{tag}] pelvis={np.round(p, 3)} yaw={yaw:+.1f}deg |q|max={np.abs(q).max():.2f}")

        time.sleep(1.0)
        probe("before init")
        # The operator's sequence: release the gantry and start the policy right after spawning.
        sim.virtual_gantry.set_enable(False)
        time.sleep(0.01)
        # The sim thread may have applied one more band force between our flag flip and its
        # check; clear the external wrenches once more after it has stepped.
        sim.root_data.xfrc_applied[:] = 0.0
        _send(master, "]", gap=0.0)
        time.sleep(1.0)
        probe("1s after release")
        _send(master, "=")
        _send(master, "w" * args.speed_presses)
        t0 = time.time()
        stopped = False
        yaw_cmd = 0.0
        while time.time() - t0 < args.walk_seconds:
            _drain(master, log)
            p = sim.robot_root_states[0, :3].detach().cpu().numpy().astype(float)
            quat = sim.robot_root_states[0, 3:7].detach().cpu().numpy().astype(float)  # xyzw
            yaw = float(
                np.degrees(
                    np.arctan2(2 * (quat[3] * quat[2] + quat[0] * quat[1]), 1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
                )
            )
            track.append((time.time() - t0, *p, _ground_below(sim, p), yaw))
            if not stopped and p[0] > args.stop_x:
                _send(master, "z", gap=0.0)  # on the far platform: stand instead of walking off its edge
                stopped = True
            elif not stopped and not args.no_heading_hold:
                # Pure pursuit to the centreline, then a P yaw-rate command quantized to the 0.1 key step.
                heading = np.arctan2(-0.8 * (p[1] + 2.0), 1.0)
                err = (heading - np.radians(yaw) + np.pi) % (2 * np.pi) - np.pi
                want = float(np.clip(np.round(1.5 * err, 1), -0.5, 0.5))
                steps = int(round((want - yaw_cmd) / 0.1))
                if steps:
                    # Spaced out: the terminal key listener merges fast repeats of one key into a hold.
                    _send(master, ("e" if steps > 0 else "q") * abs(steps), gap=0.15)
                    yaw_cmd += 0.1 * steps
            time.sleep(0.2)
        xs = np.array([t[1] for t in track])
        height = np.array([t[3] - t[4] for t in track])  # pelvis above the course surface below it
        fall_at = int(np.argmax(height < args.fall_z)) if np.any(height < args.fall_z) else None
        fell = fall_at is not None
        ok = (xs.max() > args.success_x) and not fell
        for t, x, y, z, g, yaw in track[:: max(1, len(track) // 25)]:
            print(f"t={t:5.1f}s  pelvis x={x:+.2f} y={y:+.2f} z={z:.2f} ground={g:.2f} yaw={yaw:+.1f}deg")
        where = (
            ""
            if fall_at is None
            else f"  fell at t={track[fall_at][0]:.1f}s x={track[fall_at][1]:+.2f} y={track[fall_at][2]:+.2f}"
        )
        verdict = "PASS" if ok else "FAIL"
        print(f"max x={xs.max():.2f}  min height={height.min():.2f}{where}  -> {verdict}")
        if args.json:
            import json

            with open(args.json, "w") as f:
                json.dump({"ok": bool(ok), "fall_at": fall_at, "track": track}, f)
    finally:
        _send(master, "o", gap=0.1)
        policy.terminate()
        try:
            policy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            policy.kill()
        _drain(master, log)
        log.close()
        sys.stdout.flush()
        os._exit(0 if ok else 1)  # the sim thread and its viewer-less loop have no clean stop
    return 0


if __name__ == "__main__":
    sys.exit(main())
