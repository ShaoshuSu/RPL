"""Drive the sim2sim policy through a list of waypoints.

Reads the base pose the sim sends over UDP, and sends (vx, 0, wz) to the policy, which must run with
RPL_VELOCITY_INPUT=udp (see udp_velocity_input.py). Start the policy (`]`) and switch it to walk (`=`) as usual;
this script only replaces the velocity keys.

Command limits follow the training ranges in rpl_training/config_values/command.py: |v| <= 1, |wz| <= 1,
speeds below the 0.2 m/s training deadband are treated as standing, and lateral velocity was zeroed on every
terrain, so vy is always 0. Each waypoint is (x, y, direction): direction +1 turns the robot's front toward the
waypoint and walks forward, -1 turns its back toward it and walks backward.

A sim reset (F5: sim time jumps back) restarts the route from the first waypoint.

    python -m rpl_sim2sim.waypoint_follower                       # one lap, then the lap reversed walking backward
    python -m rpl_sim2sim.waypoint_follower --return forward      # ... reversed lap walking forward instead
    python -m rpl_sim2sim.waypoint_follower --route full          # each row out and back, rows switched on the ground
    python -m rpl_sim2sim.waypoint_follower --waypoints 2.55,-2.02 0.3,-2.0,-1
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import time

from rpl_sim2sim.waypoint_udp import HOST, POSE_FORMAT, POSE_PORT, VELOCITY_FORMAT, VELOCITY_PORT

# Course in assets/scene_g1_29dof_terrains_freebase_real.xml, starting from run_sim.py --spawn 0 -2 0.8.
# Left row (y ~ -2): 4 stairs (0.15 m risers, x 1.12-2.12) up to the 0.75 m platform (x 2.12-2.88), stepping
#   stones x 3.05-5.10 in two columns (y -2.30..-2.05 / -2.00..-1.75; y -2.02 runs between them), far left
#   platform x 5.35-6.65, y -2.65..-1.35.
# Right row (y ~ -3.75): 20 deg ramp (x 1.92-3.83, ground to 0.74 m), platform x 3.80-5.10, a 0.25 m gap, far
#   right platform x 5.35-6.65, y -4.40..-3.10.
# The two far platforms are 0.45 m apart (y -3.10..-2.65): the lap turns there and steps across walking forward.
# Elsewhere the rows are 0.30 m apart at 0.75 m height with no turning room, so "full" switches rows on the ground.
START = (0.0, -2.0)
LAP = [
    (2.55, -2.02),  # up the stairs
    (6.0, -2.02),  # over the stepping stones to the far left platform
    (6.0, -3.75),  # turn to -y and step across the 0.45 m gap to the far right platform
    (4.45, -3.75),  # turn to -x, across the 0.25 m gap
    (1.0, -3.75),  # down the ramp
    START,  # on the ground back to the start
]
FULL = [
    (2.55, -2.02),  # up the stairs
    (6.0, -2.02),  # over the stepping stones to the far left platform, then turn around
    (2.55, -2.02),  # back over the stones
    (0.3, -2.0),  # down the stairs to the ground
    (1.0, -3.75),  # walk on the ground to the foot of the ramp
    (4.45, -3.75),  # up the ramp
    (6.0, -3.75),  # across the 0.25 m gap to the far right platform, then turn around
    (4.45, -3.75),  # back across the gap
    (1.0, -3.75),  # down the ramp
    START,
]
LEFT = [(2.55, -2.02), (6.0, -2.02), (2.55, -2.02), (-0.5, -2.02)]

V_MAX = 0.5  # m/s; --v-max overrides
V_MIN = 0.3  # m/s; stays above the 0.2 m/s training deadband
W_MAX = 0.6  # rad/s
K_W = 1.5  # wz per rad of heading error
TURN_IN_PLACE = math.radians(30)  # above this heading error, stop and turn
SLOW_RADIUS = 0.6  # m; slow from V_MAX to V_MIN inside this distance
REACH_RADIUS = 0.25  # m
# Crossing the 0.45 m gap between the far platforms (x 5.35-6.65, y -3.10..-2.65): at 0.5 m/s the policy steps in
# place at the edge for >10 s; at 0.8 m/s it crossed after ~1.5 s (2026-10-21). Segments whose endpoints sit on
# opposite far platforms use this speed instead of --v-max.
WIDE_GAP_V_MAX = 0.8  # m/s


def build_route(name: str, return_mode: str) -> list[tuple[float, float, int]]:
    if name == "left":
        return [(x, y, +1) for x, y in LEFT]
    if name == "full":
        return [(x, y, +1) for x, y in FULL]
    # "loop": one lap, then the same waypoints in reverse order back to the start
    back = -1 if return_mode == "backward" else +1
    return [(x, y, +1) for x, y in LAP] + [(x, y, back) for x, y in LAP[-2::-1] + [START]]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def command(
    x: float, y: float, yaw: float, goal: tuple[float, float, int], v_max: float = V_MAX
) -> tuple[float, float, float]:
    gx, gy, direction = goal
    dist = math.hypot(gx - x, gy - y)
    facing = yaw if direction > 0 else yaw + math.pi  # the side of the body that leads
    err = wrap(math.atan2(gy - y, gx - x) - facing)
    wz = max(-W_MAX, min(W_MAX, K_W * err))
    if abs(err) > TURN_IN_PLACE:
        return 0.0, wz, dist
    speed = V_MIN + (v_max - V_MIN) * min(1.0, dist / SLOW_RADIUS)
    return direction * speed * math.cos(err), wz, dist


def crosses_wide_gap(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    on_far = lambda p: 5.35 <= p[0] <= 6.65
    return on_far(a) and on_far(b) and (a[1] > -2.65) != (b[1] > -2.65) and min(a[1], b[1]) < -3.10


def parse_waypoint(text: str) -> tuple[float, float, int]:
    parts = text.split(",")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"waypoint must be X,Y or X,Y,DIR (DIR = 1 or -1): {text!r}")
    direction = int(parts[2]) if len(parts) == 3 else 1
    if direction not in (1, -1):
        raise argparse.ArgumentTypeError(f"DIR must be 1 or -1: {text!r}")
    return float(parts[0]), float(parts[1]), direction


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", choices=["loop", "full", "left"], default="loop")
    ap.add_argument("--return", dest="return_mode", choices=["backward", "forward"], default="backward",
                    help="how route 'loop' walks the reversed lap")
    ap.add_argument("--waypoints", nargs="+", type=parse_waypoint, default=None, metavar="X,Y[,DIR]",
                    help="overrides --route")
    ap.add_argument("--v-max", type=float, default=V_MAX, help="m/s, at most 1.0 (training range)")
    args = ap.parse_args()
    if not V_MIN <= args.v_max <= 1.0:
        ap.error(f"--v-max must be in [{V_MIN}, 1.0]")
    route = args.waypoints if args.waypoints else build_route(args.route, args.return_mode)
    prev = [START] + [wp[:2] for wp in route[:-1]]
    v_max = [WIDE_GAP_V_MAX if crosses_wide_gap(a, wp) else args.v_max for a, wp in zip(prev, route)]

    pose_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose_sock.bind((HOST, POSE_PORT))
    pose_sock.settimeout(1.0)
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send = lambda vx, wz: cmd_sock.sendto(struct.pack(VELOCITY_FORMAT, vx, 0.0, wz), (HOST, VELOCITY_PORT))

    print("route (x, y, direction):", flush=True)
    for k, wp in enumerate(route):
        print(f"  {k + 1:2d}  {wp}  v_max={v_max[k]:.1f}", flush=True)
    i, last_t, last_print = 0, None, 0.0
    try:
        while True:
            try:
                t, x, y, z, yaw = struct.unpack(POSE_FORMAT, pose_sock.recv(64))
            except socket.timeout:
                send(0.0, 0.0)
                print("no pose from the sim for 1 s (is run_sim.py running?)", flush=True)
                continue
            if last_t is not None and t < last_t - 0.1:
                i = 0
                print("sim reset detected: restarting the route", flush=True)
            last_t = t
            if i >= len(route):
                send(0.0, 0.0)
                continue
            vx, wz, dist = command(x, y, yaw, route[i], v_max[i])
            if dist < REACH_RADIUS:
                print(f"reached waypoint {i + 1}/{len(route)} {route[i]} at t={t:.1f}s", flush=True)
                i += 1
                if i == len(route):
                    print("route done: holding zero velocity (Ctrl-C to quit)", flush=True)
                continue
            send(vx, wz)
            if time.monotonic() - last_print > 0.5:
                last_print = time.monotonic()
                print(
                    f"t={t:6.1f}s  pos=({x:+.2f},{y:+.2f},{z:.2f}) yaw={math.degrees(yaw):+6.1f}  "
                    f"-> wp {i + 1}/{len(route)} {route[i]} dist={dist:.2f}  cmd vx={vx:+.2f} wz={wz:+.2f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        send(0.0, 0.0)


if __name__ == "__main__":
    main()
