"""Simulator presets for RPL / FALCON-Terrains."""

from dataclasses import replace

from holosoma.config_types.simulator import PhysxConfig
from holosoma.config_values.simulator import isaacgym

# IsaacGym with the PhysX settings used by FAR-FALCON-Terrains (humanoidverse).
isaacgym_falcon = replace(
    isaacgym,
    config=replace(
        isaacgym.config,
        sim=replace(
            isaacgym.config.sim,
            physx=PhysxConfig(
                solver_type=1,
                num_position_iterations=4,
                num_velocity_iterations=0,
                num_threads=4,
                bounce_threshold_velocity=0.5,
                contact_offset=0.01,
                rest_offset=0.0,
                max_depenetration_velocity=1.0,
                default_buffer_size_multiplier=5,
                contact_collection=2,
            ),
        ),
    ),
)
