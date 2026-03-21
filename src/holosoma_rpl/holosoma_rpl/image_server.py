"""Image server for streaming depth images to policy via shared memory.

Supports both MuJoCo renderer (sim2sim) and ZED cameras (sim2real).
"""

from __future__ import annotations

import time
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from multiprocessing import shared_memory
from pathlib import Path
from queue import Queue
from typing import TypedDict

import cv2
import numpy as np
from holosoma.utils.rate import RateLimiter

from holosoma_rpl.image_server_config import (
    DEFAULTS,
    ImageSaverConfig,
    ImageServerConfig,
    ImageVisualizerConfig,
)
from holosoma_rpl.mujoco_renderer import MujocoRendererWrapper
from holosoma_rpl.sensors_utils import _prepare_depth_for_visualization
from holosoma_rpl.zed import ZedCamerasConfig, ZedCamerasWrapper


class FrameBundle(TypedDict, total=False):
    rgb: dict[str, np.ndarray | None]
    depth: dict[str, np.ndarray | None]
    calibration: dict[str, dict[str, np.ndarray]]


class TimeProfiler:
    """Profiler for tracking time statistics."""

    def __init__(self):
        self.times = []
        self.count = 0
        self.total_time = 0.0
        self.min_time = float("inf")
        self.max_time = 0.0

    def record(self, elapsed_ms: float):
        self.times.append(elapsed_ms)
        self.count += 1
        self.total_time += elapsed_ms
        self.min_time = min(self.min_time, elapsed_ms)
        self.max_time = max(self.max_time, elapsed_ms)

    @contextmanager
    def measure(self):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.record(elapsed_ms)

    def get_stats(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        avg_time = self.total_time / self.count
        return {
            "count": self.count,
            "avg_ms": avg_time,
            "min_ms": self.min_time,
            "max_ms": self.max_time,
            "total_ms": self.total_time,
        }

    def reset(self):
        self.times.clear()
        self.count = 0
        self.total_time = 0.0
        self.min_time = float("inf")
        self.max_time = 0.0


class ImageSaver:
    """Handles image saving logic with background worker threads."""

    def __init__(self, config: ImageSaverConfig, near_clip: float, far_clip: float):
        self.config = config
        self.image_root_dir = config.image_root_dir
        self.near_clip = near_clip
        self.far_clip = far_clip

        session_dir = (
            Path(config.image_root_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        session_dir.mkdir(exist_ok=True, parents=True)
        self.save_images_dir = session_dir
        print(
            f"[Image Saver] Initialized save images directory: {self.save_images_dir}"
        )

        self.profiler = TimeProfiler()

        self.save_queue: Queue = Queue(maxsize=max(0, config.save_queue_maxsize))
        self.created_camera_channel_dirs: set[tuple[str, str]] = set()
        self._dirs_lock = threading.Lock()

        self.save_threads = []
        for i in range(config.save_workers):
            thread = threading.Thread(
                target=self._save_images_worker, daemon=True, name=f"SaveWorker-{i}"
            )
            thread.start()
            self.save_threads.append(thread)

    def _ensure_camera_dirs(self, names: list[str], channels: list[str]):
        with self._dirs_lock:
            for cam_name in names:
                for channel in channels:
                    camera_channel = (cam_name, channel)
                    if camera_channel in self.created_camera_channel_dirs:
                        continue
                    (self.save_images_dir / cam_name / channel).mkdir(
                        exist_ok=True, parents=True
                    )
                    self.created_camera_channel_dirs.add(camera_channel)

    def save_calibration(self, calibration_by_camera: dict[str, dict[str, np.ndarray]]):
        """Save camera intrinsics/extrinsics under each camera calibration directory."""
        if not calibration_by_camera:
            return

        for cam_name, calibration in calibration_by_camera.items():
            calibration_dir = self.save_images_dir / cam_name / "calibration"
            calibration_dir.mkdir(exist_ok=True, parents=True)

            intrinsics = calibration.get("intrinsics")
            extrinsics = calibration.get("extrinsics")

            if intrinsics is not None:
                np.save(calibration_dir / "intrinsics.npy", intrinsics)
            if extrinsics is not None:
                np.save(calibration_dir / "extrinsics.npy", extrinsics)

        print(
            f"[Image Saver] Saved calibration for cameras: {list(calibration_by_camera.keys())}"
        )

    def _save_images_worker(self):
        while True:
            try:
                item = self.save_queue.get()
                if item is None:
                    break

                raw_images, step_count, timestamp = item
                rgb_by_camera = raw_images.get("rgb", {})
                names = list(rgb_by_camera.keys())
                channels = [
                    channel for channel in ("rgb", "depth") if raw_images.get(channel)
                ]
                self._ensure_camera_dirs(names, channels)

                for cam_name in names:
                    for channel in channels:
                        channel_frames = raw_images.get(channel, {})
                        frame = channel_frames.get(cam_name)
                        if frame is None:
                            continue

                        save_path = (
                            self.save_images_dir
                            / cam_name
                            / channel
                            / f"{cam_name}_{step_count}_{timestamp}.png"
                        )
                        frame_for_png = (
                            _prepare_depth_for_visualization(
                                frame, near_clip=self.near_clip, far_clip=self.far_clip
                            )
                            if channel == "depth"
                            else frame
                        )
                        cv2.imwrite(str(save_path), frame_for_png)

                self.save_queue.task_done()
            except Exception as e:
                print(f"[Image Saver] Error in save_images_worker: {e}")

    def save(self, raw_images: FrameBundle, step_count: int):
        with self.profiler.measure():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            images_copy: FrameBundle = {}
            for channel in ("rgb", "depth"):
                channel_frames = raw_images.get(channel)
                if not channel_frames:
                    continue
                images_copy[channel] = {
                    name: (frame.copy() if frame is not None else None)
                    for name, frame in channel_frames.items()
                }
            self.save_queue.put((images_copy, step_count, timestamp))

    def get_stats(self) -> dict[str, float]:
        stats = self.profiler.get_stats()
        stats["queue_size"] = self.save_queue.qsize()
        return stats

    def shutdown(self):
        for _ in self.save_threads:
            self.save_queue.put(None)
        for thread in self.save_threads:
            thread.join()


class ImageVisualizer:
    """Handles image visualization logic."""

    def __init__(self, config: ImageVisualizerConfig):
        self.near_clip = config.near_clip
        self.far_clip = config.far_clip
        self.scale = config.scale

    def _display_frame(self, name: str, frame: np.ndarray) -> bool:
        if self.scale < 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (max(1, int(w * self.scale)), max(1, int(h * self.scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        cv2.imshow(f"Image Server Stream {name}", frame)
        return cv2.waitKey(1) & 0xFF == ord("q")

    def _depth_or_placeholder(
        self,
        depth: np.ndarray | None,
        height: int,
        width: int,
        label: str,
    ) -> np.ndarray:
        if depth is None:
            vis = np.zeros((height, width), dtype=np.uint8)
            label = f"{label} not available"
        else:
            vis = _prepare_depth_for_visualization(
                depth, near_clip=self.near_clip, far_clip=self.far_clip
            )

        cv2.putText(
            vis,
            label,
            (10, min(height - 10, 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            255,
            2,
            cv2.LINE_AA,
        )
        return vis

    def visualize(self, raw_image: FrameBundle):
        rgb_by_camera = raw_image.get("rgb", {})
        names = list(rgb_by_camera.keys())
        if not names:
            return

        # Show depth
        depth_rows = []
        for cam_name in names:
            depth = raw_image.get("depth", {}).get(cam_name)
            rgb_frame = rgb_by_camera.get(cam_name)
            if rgb_frame is None:
                continue
            h, w = rgb_frame.shape[:2]
            depth_vis = self._depth_or_placeholder(depth, h, w, f"{cam_name}: depth")
            depth_rows.append(depth_vis)

        if depth_rows:
            concatenated = np.concatenate(depth_rows, axis=0)
            self._display_frame("depth", concatenated)

        # Show RGB
        rgb_frames = [
            rgb_by_camera[name] for name in names if rgb_by_camera.get(name) is not None
        ]
        if rgb_frames:
            concatenated_rgb = np.concatenate(rgb_frames, axis=0)
            self._display_frame("rgb", concatenated_rgb)


class ImageServer:
    def __init__(
        self,
        camera_wrapper: MujocoRendererWrapper | ZedCamerasWrapper,
        cfg: ImageServerConfig,
    ):
        self.cfg: ImageServerConfig = cfg
        self.camera_wrapper: ZedCamerasWrapper | MujocoRendererWrapper = camera_wrapper

        # Initialize shared memory
        self._init_shared_memory()

        # Initialize delay buffer
        self.num_delay_frames = cfg.num_delay_frames
        if self.num_delay_frames > 0:
            self.delay_buffer: deque[tuple[int, np.ndarray]] = deque(
                maxlen=self.num_delay_frames + 1
            )
            print(
                f"[Image Server] Initialized delay buffer with {self.num_delay_frames} delay frames"
            )
        else:
            self.delay_buffer = None

        # Initialize profiler
        self.capture_profiler = TimeProfiler()

        # Initialize image saver and visualizer
        if cfg.save_images:
            self.image_saver = ImageSaver(
                cfg.image_saver_config, near_clip=cfg.near_clip, far_clip=cfg.far_clip
            )
            self._save_camera_calibration()
        else:
            self.image_saver = None

        if cfg.visualize_images:
            self.image_visualizer = ImageVisualizer(config=cfg.image_visualizer_config)
        else:
            self.image_visualizer = None

    def _save_camera_calibration(self):
        """Save camera calibration right after ImageSaver initialization."""
        cameras = getattr(self.camera_wrapper, "cameras", None)
        if cameras is None:
            return

        calibration_by_camera: dict[str, dict[str, np.ndarray]] = {}
        for cam_name, camera in cameras.items():
            calibration = getattr(camera, "calibration", None)
            if calibration is None:
                continue
            calibration_by_camera[cam_name] = calibration

        if not calibration_by_camera:
            return

        self.image_saver.save_calibration(calibration_by_camera)

    def _init_shared_memory(self):
        img_shm_name = "depth_img_shm"
        expected_shape = [
            self.camera_wrapper.num_cameras,
            1,
            self.cfg.resized_height,
            self.cfg.resized_width,
        ]
        dtype = np.float32

        memory_size = int(np.prod(expected_shape)) * np.dtype(dtype).itemsize
        try:
            self.image_shm = shared_memory.SharedMemory(
                create=True, size=memory_size, name=img_shm_name
            )
            print(f"[Image Server] Created new shared memory: {img_shm_name}")
        except FileExistsError:
            self.image_shm = shared_memory.SharedMemory(name=img_shm_name)
            print(f"[Image Server] Connected to existing shared memory: {img_shm_name}")

        self.img_array = np.ndarray(
            expected_shape, dtype=dtype, buffer=self.image_shm.buf
        )
        print(f"[Image Server] Shared memory: shape={expected_shape}, dtype={dtype}")
        print("ImageServer initialized")

    def _resize_clip_expand_transpose(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(
            frame, (self.cfg.resized_width, self.cfg.resized_height), cv2.INTER_CUBIC
        )
        frame = np.clip(frame, self.cfg.near_clip, self.cfg.far_clip)
        frame = (frame - self.cfg.near_clip) / (
            self.cfg.far_clip - self.cfg.near_clip
        ) - 0.5
        frame = np.expand_dims(frame, axis=0)
        return frame

    def send_process(self):
        """Main loop: render depth at 10Hz and write to shared memory."""
        render_frequency = 10
        rate_limiter = RateLimiter(render_frequency)

        step_count = 0
        while True:
            with self.capture_profiler.measure():
                all_frames: FrameBundle = dict(self.camera_wrapper.get_frames())

            depth_for_policy = all_frames[self.cfg.depth_source]
            depth_for_policy = [
                self._resize_clip_expand_transpose(frame)
                for frame in depth_for_policy.values()
            ]
            full_depth_for_policy = np.stack(depth_for_policy, axis=0)

            # Delay buffer
            if self.delay_buffer is not None:
                self.delay_buffer.append((step_count, full_depth_for_policy.copy()))
                if len(self.delay_buffer) <= self.num_delay_frames:
                    rate_limiter.sleep()
                    step_count += 1
                    continue
                _delayed_step_count, delayed_image = self.delay_buffer[0]
            else:
                delayed_image = full_depth_for_policy

            # Copy to shared memory
            try:
                np.copyto(self.img_array, delayed_image)
            except Exception as e:
                print(f"[Image Server] Failed to copy to shared memory: {e}")
                continue

            # Save and visualize
            if self.image_saver is not None:
                self.image_saver.save(all_frames, step_count)

            if self.image_visualizer is not None:
                self.image_visualizer.visualize(all_frames)

            if step_count % 50 == 0:
                print(f"[Image Server] Rate limiter stats: {rate_limiter.get_stats()}")
                rate_limiter.reset()
                print(
                    f"[Image Server] capture stats: {self.capture_profiler.get_stats()}"
                )

            rate_limiter.sleep()
            step_count += 1


if __name__ == "__main__":
    import tyro
    from typing_extensions import Annotated

    import holosoma_rpl.image_server_config

    ImageServerCliConfig = Annotated[
        ImageServerConfig,
        tyro.conf.arg(constructor=tyro.extras.subcommand_type_from_defaults(DEFAULTS)),
    ]

    cfg = tyro.cli(ImageServerCliConfig, default=holosoma_rpl.image_server_config.real)

    # Set ZED depth mode based on config toggle.
    depth_mode = "NEURAL" if cfg.enable_zed_depth_prediction else "NONE"
    zed_cfg = ZedCamerasConfig(
        terms={
            name: replace(camera_cfg, depth_mode=depth_mode)
            for name, camera_cfg in ZedCamerasConfig().terms.items()
        }
    )

    image_server = ImageServer(ZedCamerasWrapper(zed_cfg), cfg)
    thread = threading.Thread(target=image_server.send_process)
    thread.start()
    time.sleep(2)
    thread.join()
