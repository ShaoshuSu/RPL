"""Runtime patches for holosoma to add depth cameras and image server support."""

import os
from threading import Thread
from xml.etree import ElementTree as ET

from loguru import logger


DEPTH_CAMERAS = [
    {
        "name": "cam_front_depth",
        "parent_body": "torso_link",
        "pos": "0.135 0.06 0.02",
        "euler": "0 -0.2618 -1.57",
        "fovy": "69.00",
    },
    {
        "name": "cam_back_depth",
        "parent_body": "torso_link",
        "pos": "-0.125 -0.06 0.02",
        "euler": "0 0.2618 1.57",
        "fovy": "69.00",
    },
]


def _inject_cameras_into_xml(xml_path):
    """Read robot XML, inject depth cameras, return modified XML string and assets dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for cam in DEPTH_CAMERAS:
        # Find the parent body element
        for body in root.iter("body"):
            if body.get("name") == cam["parent_body"]:
                ET.SubElement(
                    body,
                    "camera",
                    {
                        "name": cam["name"],
                        "pos": cam["pos"],
                        "euler": cam["euler"],
                        "fovy": cam["fovy"],
                    },
                )
                logger.info(
                    f"Injected camera '{cam['name']}' into body '{cam['parent_body']}'"
                )
                break
        else:
            logger.warning(
                f"Body '{cam['parent_body']}' not found for camera '{cam['name']}'"
            )

    xml_string = ET.tostring(root, encoding="unicode")

    # Load mesh assets. The XML has <compiler meshdir="./meshes/"/>,
    # so MjSpec.from_string expects asset keys without the meshdir prefix.
    xml_dir = os.path.dirname(xml_path)
    assets = {}
    meshes_dir = os.path.join(xml_dir, "meshes")
    if os.path.isdir(meshes_dir):
        for fname in os.listdir(meshes_dir):
            fpath = os.path.join(meshes_dir, fname)
            if os.path.isfile(fpath):
                with open(fpath, "rb") as f:
                    assets[fname] = f.read()

    return xml_string, assets


def _patch_scene_manager_add_robot(original_add_robot):
    """Wrap MujocoSceneManager.add_robot to inject depth cameras into the XML."""

    def patched_add_robot(
        self, terrain_state, robot_config, xml_filter=None, prefix="robot_"
    ):
        import mujoco

        from holosoma.utils.module_utils import get_holosoma_root

        asset_root = robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        robot_xml_path = os.path.join(asset_root, robot_config.asset.xml_file)

        logger.info(f"Adding robot from: {robot_xml_path} with prefix: {prefix}")
        self.robot_model_path = robot_xml_path

        # Inject cameras: write temp XML in same dir (so mesh paths resolve)
        import tempfile

        xml_string, _assets = _inject_cameras_into_xml(robot_xml_path)
        xml_dir = os.path.dirname(robot_xml_path)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=xml_dir)
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(xml_string)
            robot_spec = mujoco.MjSpec.from_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        if xml_filter and getattr(xml_filter, "enable", False):
            robot_spec = self._filter_robot_worldbody(robot_spec, xml_filter)

        if hasattr(terrain_state, "geom") and terrain_state.geom:
            self._apply_collision_settings(robot_spec, robot_config)

        robot_pos = [0, 0, 0.0]
        robot_rot = [1, 0, 0, 0]
        site = self.world_spec.worldbody.add_site(pos=robot_pos, quat=robot_rot)
        self.world_spec.attach(robot_spec, site=site, prefix=prefix)

        self.robot_prefix = prefix

    return patched_add_robot


def _patch_direct_simulation_initialize(original_initialize):
    """Wrap DirectSimulation.initialize to add ImageServer + camera thread."""

    def patched_initialize(self):
        original_initialize(self)

        from holosoma_rpl.image_server import ImageServer
        from holosoma_rpl.mujoco_renderer import MujocoRendererWrapper

        logger.info("Setting up image server (holosoma-extension-rpl)")
        self.image_server = ImageServer(
            camera_wrapper=MujocoRendererWrapper(self.simulator),
            cfg=self.config.image_server,
        )
        self.cam_thread = Thread(target=self.image_server.send_process)
        self.cam_thread.daemon = True

        self.sim_thread = Thread(target=self.run)

    return patched_initialize


def apply_runtime_patches():
    """Apply runtime-only patches."""
    from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager
    from holosoma.utils.sim_utils import DirectSimulation

    MujocoSceneManager.add_robot = _patch_scene_manager_add_robot(
        MujocoSceneManager.add_robot
    )

    DirectSimulation.initialize = _patch_direct_simulation_initialize(
        DirectSimulation.initialize
    )

    logger.info("holosoma-extension-rpl runtime patches applied")
