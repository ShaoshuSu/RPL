"""Extended RunSimConfig with image server and additional terrain configs."""

from dataclasses import dataclass

import tyro
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.config_types.terrain import TerrainManagerCfg
from typing_extensions import Annotated

from holosoma_rpl.config_values.terrain import DEFAULTS as TERRAIN_DEFAULTS
from holosoma_rpl.image_server_config import (
    DEFAULTS as IMAGE_SERVER_DEFAULTS,
    ImageServerConfig,
)


@dataclass(frozen=True)
class RplRunSimConfig(RunSimConfig):
    """RunSimConfig extended with image server and additional terrain configs.

    Usage:
        python -m holosoma_rpl.run_sim robot:g1-29dof terrain:terrain_locomotion_stairs_and_slope_eval
    """

    terrain: Annotated[
        TerrainManagerCfg,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(TERRAIN_DEFAULTS)
        ),
    ] = TERRAIN_DEFAULTS["terrain_locomotion_plane"]

    image_server: Annotated[
        ImageServerConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(IMAGE_SERVER_DEFAULTS)
        ),
    ] = IMAGE_SERVER_DEFAULTS["mujoco"]
