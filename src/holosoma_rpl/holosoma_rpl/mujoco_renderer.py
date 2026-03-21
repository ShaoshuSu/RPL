"""MuJoCo depth/RGB renderer wrapper for the image server.

Renders depth and RGB frames from named cameras defined in the MuJoCo XML.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from holosoma.simulator.mujoco.mujoco import MuJoCo


class MujocoRendererWrapper:
    def __init__(self, simulator: MuJoCo) -> None:
        # TODO: get height/width of renderer from config
        self._height = 135
        self._width = 240
        self.camera_names = [
            "robot_cam_front_depth",
            "robot_cam_back_depth",
        ]

        self.num_cameras = 2

        self._simulator: MuJoCo = simulator

        self._depth_renderer: mujoco.Renderer | None = None
        self._depth_renderer_thread_id: int | None = None

        self._rgb_renderer: mujoco.Renderer | None = None
        self._rgb_renderer_thread_id: int | None = None

        self.cameras = None

    @property
    def depth_renderer(self) -> mujoco.Renderer:
        """Lazy initialization of depth renderer on the rendering thread."""
        current_thread_id = threading.get_ident()
        if (
            self._depth_renderer is None
            or self._depth_renderer_thread_id != current_thread_id
        ):
            self._depth_renderer = mujoco.Renderer(
                self._simulator.root_model, height=self._height, width=self._width
            )
            self._depth_renderer.enable_depth_rendering()
            self._depth_renderer_thread_id = current_thread_id
        return self._depth_renderer

    @property
    def rgb_renderer(self) -> mujoco.Renderer:
        """Lazy initialization of rgb renderer on the rendering thread."""
        current_thread_id = threading.get_ident()
        if (
            self._rgb_renderer is None
            or self._rgb_renderer_thread_id != current_thread_id
        ):
            self._rgb_renderer = mujoco.Renderer(
                self._simulator.root_model, height=self._height, width=self._width
            )
            self._rgb_renderer_thread_id = current_thread_id
        return self._rgb_renderer

    def get_frames(self) -> dict[str, dict[str, np.ndarray]]:
        """Get depth and rgb frames from the renderer."""
        world_id = getattr(self._simulator, "current_world_id", 0)
        render_data = self._simulator.backend.get_render_data(world_id=world_id)

        depth_frames = {}
        rgb_frames = {}

        for camera_name in self.camera_names:
            self.depth_renderer.update_scene(render_data, camera=camera_name)
            depth_frames[camera_name] = self.depth_renderer.render()

            self.rgb_renderer.update_scene(render_data, camera=camera_name)
            rgb_frames[camera_name] = self.rgb_renderer.render()

        return {"depth": depth_frames, "rgb": rgb_frames}
