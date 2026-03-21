"""Extended terrain configurations."""

from holosoma.config_types.terrain import (
    MeshType,
    SpawnCfg,
    TerrainManagerCfg,
    TerrainTermCfg,
)
from holosoma.config_values.terrain import DEFAULTS as HOLOSOMA_TERRAIN_DEFAULTS

terrain_locomotion_stairs = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.TRIMESH,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        border_size=1,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=2,
        num_cols=2,
        terrain_config={
            "smooth_stairs": 1.0,
        },
        step_width_range=[0.25, 0.25],
        fixed_step_height=0.15,
        max_slope=0.3,
        slope_treshold=0.75,
        spawn=SpawnCfg(
            randomize_tiles=False,
            xy_offset_range=0.0,
            query_terrain_height=True,
            use_grid_sampling=True,
            grid_size=20,
            grid_spacing=0.01,
        ),
    )
)

# Combined defaults: holosoma defaults + extension defaults
DEFAULTS = {
    **HOLOSOMA_TERRAIN_DEFAULTS,
    "terrain_locomotion_stairs": terrain_locomotion_stairs,
}
