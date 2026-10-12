"""Terrain presets for RPL / FALCON-Terrains."""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.terrain import MeshType, SpawnCfg, TerrainManagerCfg, TerrainTermCfg

# General multi-terrain preset mirroring FAR-FALCON's ``falcon_hm_multi_terrain``
# terrain block.  Registers the five directional/stepping terrain types ported
# from FAR-FALCON-Terrains so a single preset covers slopes (up/down), stairs
# (up/down), and 2-column stepping stones.  All types start with equal (0.2)
# proportion; override ``terrain_config`` from the CLI to focus training on a
# subset (e.g. slopes-only, stairs-only, stones-only).  ``max_slope`` and
# ``step_width_range`` are set generously so per-scenario CLI overrides work
# without changing the preset.
terrain_locomotion_multi = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        mesh_type=MeshType.TRIMESH,
        # Match FAR-FALCON's ``terrain_locomotion.yaml`` base:
        # ``horizontal_scale=0.05`` (not 0.1) gives 4x denser heightfield cells.
        # Critical for stairs (step widths 0.30 m = 6 cells instead of 3) and
        # for stepping-stones (inter-stone gaps of 0.05 m round to 1 cell, not 0).
        # ``border_size=25`` matches the yaml default.
        horizontal_scale=0.05,
        vertical_scale=0.005,
        border_size=25,
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        terrain_move_down_ratio=0.5,
        terrain_move_up_ratio=0.3125,
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=10,
        num_cols=20,
        max_init_terrain_level=2,
        terrain_config={
            # ``slopes`` mirrors FAR-FALCON-Terrains' 50/50 mix of pyramid
            # ``slope_up`` and ``slope_down`` rolled up under a single bucket
            # (one direction is sampled per tile inside the terrain func).
            "slopes": 0.2,
            "stairs_up": 0.25,
            "stairs_down": 0.25,
            "stepping_stones_columns": 0.3,
        },
        max_slope=0.7,
        step_width_range=[0.30, 0.35],
        stairs_offset=0.01,
        stairs_ratio=0.25,
        random_step_height_range=[-0.01, 0.03],
        stone_size_range=[0.25, 0.40],
        stone_distance_row_range=[0.025, 0.075],
        stone_distance_col_range=[0.05, 0.7],
        stone_height_range=[0.0, 0.05],
        # FAR-FALCON's base ``terrain_locomotion.yaml`` sets ``depth_range: [0.4, 2.5]``,
        # which overrides ``stepping_stone_terrain_func``'s built-in default (0.4-0.6).
        # Match that wider range so stones have varied pit depths like FALCON.
        stone_depth_range=[0.4, 2.5],
        # Mirrors FAR-FALCON's ``not_measure_heights_mask: [2, 7]``. Stepping
        # stones' ray-cast base/feet heights would read from pit floors, so
        # fall back to raw world-Z for envs on this terrain type. FALCON also
        # lists ``gap`` (index 7); we don't have that terrain in this preset.
        raw_base_height_terrain_mask=["stepping_stones_columns"],
        slope_treshold=0.75,
        enable_height_scan=True,
        spawn=SpawnCfg(
            randomize_tiles=True,
            xy_offset_range=0.75,
            # Stepping stones get the narrow y-jitter so spawns stay on the row pair.
            # Listed types keep the full symmetric range (±0.75).
            xy_offset_terrain_mask=["slopes", "stairs_up", "stairs_down"],
            xy_offset_range_x_masked=[-0.75, 0.75],
            xy_offset_range_y_masked=[-0.25, 0.25],
        ),
    )
)


# FAR-FALCON-Terrains ``terrain_locomotion`` as resolved by ``+exp=falcon_hm_dagger_moe``:
# stepping_stone 0.35 / rough_slope 0.15 / stairs_up 0.25 / stairs_down 0.25
# (same 7/3/5/5 column split of the 20 columns), ``max_slope=0.75``,
# ``step_width_range=[0.25, 0.30]``, rough-slope noise, and the teacher
# heightmap sampled from the raw heightfield like FALCON's ``_get_heights``.
terrain_locomotion_multi_falcon = replace(
    terrain_locomotion_multi,
    terrain_term=replace(
        terrain_locomotion_multi.terrain_term,
        # Order matches FALCON's non-zero ``terrain_types`` (stepping_stone,
        # rough_slope, stairs_up, stairs_down): identical column layout, and
        # ``terrain_type_idx`` 0..3 lines up with the MoE teacher's expert order.
        terrain_config={
            "stepping_stones_columns": 0.35,
            "slopes": 0.15,
            "stairs_up": 0.25,
            "stairs_down": 0.25,
        },
        max_slope=0.75,
        step_width_range=[0.25, 0.30],
        slopes_roughness=True,
        height_scan_mode="falcon",
        body_height_mode="falcon",
    ),
)


# FAR-FALCON-Terrains ``terrain_locomotion`` as resolved by ``+exp=falcon_hm_multi_terrain``
# (RL experts): heightmap and base / feet heights sampled from the raw heightfield like
# FALCON, and the directional ``slope_up`` / ``slope_down`` types (FALCON fixes the slope
# direction per column, Holosoma's ``slopes`` bucket draws it per tile). Every type is
# listed so each expert selects its mix on the CLI, e.g. slopes
# ``--terrain.terrain-term.terrain-config.slope-up=0.5 ...slope-down=0.5`` with the rest 0.
# The default mix is the multi-terrain one with ``slopes`` split into up / down.
terrain_locomotion_multi_falcon_expert = replace(
    terrain_locomotion_multi,
    terrain_term=replace(
        terrain_locomotion_multi.terrain_term,
        terrain_config={
            "slopes": 0.0,
            "slope_up": 0.1,
            "slope_down": 0.1,
            "stairs_up": 0.25,
            "stairs_down": 0.25,
            "stepping_stones_columns": 0.3,
        },
        height_scan_mode="falcon",
        body_height_mode="falcon",
        spawn=replace(
            terrain_locomotion_multi.terrain_term.spawn,
            xy_offset_terrain_mask=["slopes", "slope_up", "slope_down", "stairs_up", "stairs_down"],
        ),
    ),
)


def _falcon_expert_terrain(terrain_config: dict[str, float], **overrides) -> TerrainManagerCfg:
    """Single-terrain expert mix; all of FALCON's final expert stages resolve
    ``terrain.spacing_size=1.0`` (1 m flat gaps between tiles)."""
    base = terrain_locomotion_multi_falcon_expert.terrain_term
    mix = {name: 0.0 for name in base.terrain_config}
    return replace(
        terrain_locomotion_multi_falcon_expert,
        terrain_term=replace(base, terrain_config={**mix, **terrain_config}, spacing_size=1.0, **overrides),
    )


# Per-terrain experts as in FALCON's final expert stages (wandb run args).
# stones far-wandb/falcon-29-terrains-experts-stones/lrlo8qpt: ``terrain_proportions`` stones only.
terrain_locomotion_falcon_expert_stones = _falcon_expert_terrain({"stepping_stones_columns": 1.0})
# stairs far-wandb/falcon-29-terrains-experts/h5d52hm0: ``step_width_range=[0.25,0.3]``.
terrain_locomotion_falcon_expert_stairs = _falcon_expert_terrain(
    {"stairs_up": 0.5, "stairs_down": 0.5}, step_width_range=[0.25, 0.30]
)
# slopes far-wandb/falcon-29-terrains-experts/b04ael3p: ``rough_terrain=True`` (uniform noise
# on slope_up / slope_down) and ``max_slope=0.75``.
terrain_locomotion_falcon_expert_slopes = _falcon_expert_terrain(
    {"slope_up": 0.5, "slope_down": 0.5}, slopes_roughness=True, max_slope=0.75
)
