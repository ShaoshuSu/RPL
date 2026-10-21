"""Localhost UDP packets for waypoint following in sim2sim.

sim (simulator.py) --pose--> waypoint_follower.py --velocity--> policy (UdpVelocityInput)

The policy itself never sees the base pose; only the follower does, the same way an external planner would.
"""

from __future__ import annotations

import math

HOST = "127.0.0.1"
POSE_PORT = 47201  # sim -> follower: sim_time, x, y, z, yaw
VELOCITY_PORT = 47202  # follower -> policy: vx, vy, wz

POSE_FORMAT = "<5d"
VELOCITY_FORMAT = "<3d"


def yaw_from_wxyz(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
