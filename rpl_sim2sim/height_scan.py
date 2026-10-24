"""Height map for the privileged terrain teachers, computed by the sim and sent to the policy over UDP.

Matches the teachers' ``perception_obs`` (rpl_training ``height_measurements`` in holosoma's "falcon" scan mode):
a 16 x 10 grid x = linspace(-0.8, 0.8, 16), y = linspace(-0.5, 0.5, 10) in the base yaw frame, flattened x-major
(the CNN reads it as a 16 x 10 image), each value ``pelvis_z - 0.7 - terrain_z`` clipped to [-0.5, 0.5]
(``lidar_height_offset`` 0.7 from the G1 robot config).

sim (simulator.py, run_sim.py --height-scan) --udp--> policy (teacher_policy.py)
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np

HOST = "127.0.0.1"
PORT = 47203
NUM_X, NUM_Y = 16, 10
NUM_POINTS = NUM_X * NUM_Y
GRID_X, GRID_Y = (
    a.flatten() for a in np.meshgrid(np.linspace(-0.8, 0.8, NUM_X), np.linspace(-0.5, 0.5, NUM_Y), indexing="ij")
)
LIDAR_HEIGHT_OFFSET = 0.7
CLIP = 0.5
PACKET_FORMAT = f"<d{NUM_POINTS}f"  # sim time, then the scan


class HeightScanReceiver:
    """Keeps the newest scan sent by the sim; zeros until the first packet."""

    def __init__(self, port: int = PORT):
        self.port = port
        self._lock = threading.Lock()
        self._scan = np.zeros(NUM_POINTS, dtype=np.float32)
        self._stamp: float | None = None

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((HOST, self.port))
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def _recv_loop(self) -> None:
        size = struct.calcsize(PACKET_FORMAT)
        while True:
            data, _ = self._sock.recvfrom(size + 16)
            if len(data) != size:
                continue
            scan = np.asarray(struct.unpack(PACKET_FORMAT, data)[1:], dtype=np.float32)
            with self._lock:
                self._scan = scan
                self._stamp = time.monotonic()

    def latest(self) -> np.ndarray:
        with self._lock:
            return self._scan.copy()

    def age_s(self) -> float | None:
        """Seconds since the last packet, or None before the first one."""
        with self._lock:
            return None if self._stamp is None else time.monotonic() - self._stamp
