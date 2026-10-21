"""Velocity provider that takes (vx, vy, wz) from localhost UDP, sent by waypoint_follower.py.

Implements holosoma's VelCmdProvider protocol (start / poll_velocity / zero). If no packet has arrived for
TIMEOUT_S, it reports zero velocity, so a stopped or crashed follower brings the robot to a standstill.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

from holosoma_inference.inputs.api.commands import VelCmd
from loguru import logger

from rpl_sim2sim.waypoint_udp import HOST, VELOCITY_FORMAT, VELOCITY_PORT

TIMEOUT_S = 0.5


class UdpVelocityInput:
    def __init__(self, port: int = VELOCITY_PORT):
        self.port = port
        self._lock = threading.Lock()
        self._latest = (0.0, 0.0, 0.0)
        self._stamp = 0.0
        self._timed_out = True

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((HOST, self.port))
        threading.Thread(target=self._recv_loop, daemon=True).start()
        logger.info(f"[UdpVelocityInput] listening for velocity commands on udp://{HOST}:{self.port}")

    def _recv_loop(self) -> None:
        size = struct.calcsize(VELOCITY_FORMAT)
        while True:
            data, _ = self._sock.recvfrom(64)
            if len(data) != size:
                continue
            with self._lock:
                self._latest = struct.unpack(VELOCITY_FORMAT, data)
                self._stamp = time.monotonic()

    def poll_velocity(self) -> VelCmd:
        with self._lock:
            vx, vy, wz = self._latest
            stale = time.monotonic() - self._stamp > TIMEOUT_S
        if stale != self._timed_out:
            self._timed_out = stale
            logger.info(f"[UdpVelocityInput] {'no commands for 0.5 s: zero velocity' if stale else 'receiving commands'}")
        if stale:
            return VelCmd((0.0, 0.0), 0.0)
        return VelCmd((vx, vy), wz)

    def zero(self) -> None:
        with self._lock:
            self._latest = (0.0, 0.0, 0.0)
