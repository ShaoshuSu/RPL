"""FAR-FALCON-Terrains ``sim2real/utils/image_wrapper/image_server.py``: MuJoCo depth to shared memory.

``MujocoCamera`` (gymnasium's ``MujocoRenderer``, linear depth from the z-buffer) and ``ImageServer``
(cv2 bicubic resize, clip, normalize to [-0.5, 0.5], ``depth_delay`` frames of delay, one
``[N, 1, H, W]`` float32 block), MuJoCo depth only.
"""

from __future__ import annotations

import time
from collections import deque
from multiprocessing import shared_memory

import cv2
import numpy as np
from loop_rate_limiters import RateLimiter


class MujocoCamera:
    def __init__(self, mj_model, mj_data, cam_id, img_shape):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.img_shape = img_shape
        self.cam_id = cam_id
        if cam_id == -1:
            raise ValueError("Camera not found in the model.")
        # Created lazily on the rendering thread: with MUJOCO_GL=egl the GL context is bound to the
        # thread that creates it (creating it on the main thread gives EGL_BAD_ACCESS).
        self.cam_renderer = None

    def _ensure_renderer(self):
        if self.cam_renderer is None:
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

            self.cam_renderer = MujocoRenderer(
                self.mj_model, self.mj_data, camera_id=self.cam_id, width=self.img_shape[1], height=self.img_shape[0]
            )

    def get_frame(self):
        self._ensure_renderer()
        return self._depth_2_meters(self.cam_renderer.render("depth_array"))

    def _depth_2_meters(self, depth):
        extend = self.mj_model.stat.extent
        near = self.mj_model.vis.map.znear * extend
        far = self.mj_model.vis.map.zfar * extend
        return near / (1 - depth * (1 - near / far))

    def release(self):
        if self.cam_renderer is not None:
            self.cam_renderer.close()


class ImageServer:
    def __init__(self, config, img_shm_name):
        print(f"[Image Server] Configuration: {config}")
        self.near_clip = config.get("near_clip", 0.1)
        self.far_clip = config.get("far_clip", 2.5)
        self.if_resize = config.get("if_resize", False)
        self.expected_shape = (
            config.get("resized_image_shape", [480, 640]) if self.if_resize else config.get("image_shape", [480, 640])
        )
        self.depth_delay = config.get("depth_delay", 0)
        self.depth_buffer = deque(maxlen=self.depth_delay + 1)
        print(f"[Image Server] Depth delay: {self.depth_delay} frames")

        self.cameras = []
        self.camera_names = []
        image_shape = config.get("image_shape", [480, 640])
        fps = config.get("fps", 30)
        for camera_name, cam_id in config["id_numbers"].items():
            print(f"[Image Server] Initializing {camera_name}: shape={image_shape}, ids={cam_id}")
            self.cameras.append(MujocoCamera(config["mj_model"], config["mj_data"], cam_id, image_shape))
            self.camera_names.append(camera_name)
        if not self.cameras:
            raise ValueError("No valid cameras found")

        expected_shape = [len(self.cameras), 1, self.expected_shape[0], self.expected_shape[1]]
        memory_size = int(np.prod(expected_shape)) * np.dtype(np.float32).itemsize
        try:
            self.image_shm = shared_memory.SharedMemory(create=True, size=memory_size, name=img_shm_name)
            print(f"[Image Server] Created new shared memory: {img_shm_name}")
        except FileExistsError:
            self.image_shm = shared_memory.SharedMemory(name=img_shm_name)
            print(f"[Image Server] Connected to existing shared memory: {img_shm_name}")
        self.img_array = np.ndarray(expected_shape, dtype=np.float32, buffer=self.image_shm.buf)
        print(f"[Image Server] Shared memory: shape={expected_shape}, dtype=float32")

        self.rate = RateLimiter(fps)
        self.frame_count = 0
        self.frame_times: deque = deque()

    def _post_process_frame(self, frame):
        if self.if_resize:
            frame = cv2.resize(frame, (self.expected_shape[1], self.expected_shape[0]), interpolation=cv2.INTER_CUBIC)
        frame = np.clip(frame, self.near_clip, self.far_clip)
        return (frame - self.near_clip) / (self.far_clip - self.near_clip) - 0.5

    def _close(self):
        for camera in self.cameras:
            camera.release()
        self.image_shm.close()
        print("[Image Server] Closed.")

    def send_process(self):
        try:
            while True:
                frames = []
                for camera in self.cameras:
                    frame = camera.get_frame()
                    if frame is None:
                        break
                    frame = self._post_process_frame(frame)
                    frames.append(frame[None])  # [H, W] -> [C=1, H, W]
                if len(frames) != len(self.cameras):
                    continue
                full_image = np.stack(frames, axis=0) if len(frames) > 1 else frames[0]  # [N, C, H, W]

                if self.depth_delay > 0:
                    self.depth_buffer.append(full_image.copy())
                    if len(self.depth_buffer) > self.depth_delay:
                        full_image = self.depth_buffer[-(self.depth_delay + 1)]
                    else:
                        full_image = self.depth_buffer[0]

                np.copyto(self.img_array, full_image.astype(np.float32))

                current_time = time.time()
                self.frame_times.append(current_time)
                while self.frame_times and self.frame_times[0] < current_time - 1.0:
                    self.frame_times.popleft()
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    print(f"[Image Server] FPS: {len(self.frame_times):.2f}, Frames: {self.frame_count}")
                self.rate.sleep()
        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        finally:
            self._close()
