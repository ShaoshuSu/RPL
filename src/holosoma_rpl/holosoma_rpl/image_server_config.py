"""Configuration types and defaults for the image server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ImageSaverConfig:
    """Configuration for image saver."""

    image_root_dir: str = "image_server_images"
    save_queue_maxsize: int = 0
    save_workers: int = 2


@dataclass(frozen=True)
class ImageVisualizerConfig:
    """Configuration for image visualization."""

    near_clip: float = 0.1
    far_clip: float = 2.0
    scale: float = 1.0


@dataclass(frozen=True)
class ImageServerConfig:
    """Configuration for image server."""

    near_clip: float = 0.1
    far_clip: float = 2.0

    resized_height: int = 27
    resized_width: int = 48

    enable_zed_depth_prediction: bool = True
    """If True, use ZED depth stream."""

    depth_source: Literal["depth"] = "depth"
    """Depth source to send to policy shared memory."""

    visualize_images: bool = False
    image_visualizer_config: ImageVisualizerConfig = field(
        default_factory=ImageVisualizerConfig
    )

    save_images: bool = True
    image_saver_config: ImageSaverConfig = field(default_factory=ImageSaverConfig)

    num_delay_frames: int = 0
    """Number of frames to delay before sending to shared memory."""


# Default configurations
mujoco = ImageServerConfig(
    enable_zed_depth_prediction=True,
    depth_source="depth",
    save_images=True,
)

real = ImageServerConfig(
    enable_zed_depth_prediction=True,
    depth_source="depth",
    visualize_images=False,
)

real_verbose = ImageServerConfig(
    enable_zed_depth_prediction=True,
    depth_source="depth",
    visualize_images=True,
    save_images=True,
)

DEFAULTS = {
    "mujoco": mujoco,
    "real": real,
    "real_verbose": real_verbose,
}
