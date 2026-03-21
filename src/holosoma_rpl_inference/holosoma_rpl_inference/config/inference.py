"""Inference config preset for loco-manip stand-height-waist policy."""

from holosoma_inference.config.config_types.inference import InferenceConfig

from holosoma_rpl_inference.config.observation import (
    loco_manip_stand_height_waist as obs_config,
)
from holosoma_rpl_inference.config.robot import (
    g1_29dof_loco_manip_stand_height_waist as robot_config,
)
from holosoma_rpl_inference.config.task import (
    loco_manip_stand_height_waist as task_config,
)

g1_29dof_loco_manip_stand_height_waist = InferenceConfig(
    robot=robot_config,
    observation=obs_config,
    task=task_config,
)
