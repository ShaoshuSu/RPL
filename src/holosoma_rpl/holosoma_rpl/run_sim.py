#!/usr/bin/env python3
"""
Simulation Runner with depth camera support.

Uses RplRunSimConfig which extends holosoma's RunSimConfig with:
- Image server for depth streaming via shared memory
- Additional terrain configs (stairs_and_slope_eval)
- Depth cameras injected into robot model at runtime
"""

import dataclasses
import sys
import traceback

import tyro
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.sim_utils import DirectSimulation, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG
from loguru import logger

from holosoma_rpl.config_types.run_sim import RplRunSimConfig
from holosoma_rpl.patches import apply_runtime_patches


def run_simulation(config):
    """Run simulation with depth camera and image server."""
    # Auto-set device for GPU-accelerated backends
    if config.device == "cpu":
        if hasattr(config.simulator.config, "mujoco_backend"):
            from holosoma.config_types.simulator import MujocoBackend

            if config.simulator.config.mujoco_backend == MujocoBackend.WARP:
                logger.info(
                    "Auto-detected MuJoCo Warp backend - setting device to cuda:0"
                )
                config = dataclasses.replace(config, device="cuda:0")

    config = dataclasses.replace(config, device=config.device)

    logger.info("Starting Holosoma Direct Simulation (with depth cameras)...")
    logger.info(f"Robot: {config.robot.asset.robot_type}")
    logger.info(f"Simulator: {config.simulator._target_}")
    logger.info(
        f"Terrain: {config.terrain.terrain_term.mesh_type} ({config.terrain.terrain_term.func})"
    )

    try:
        env, device, simulation_app = setup_simulation_environment(
            config, device=config.device
        )

        with DirectSimulation(config, env, device, simulation_app) as sim:
            logger.info("Starting simulation thread")
            sim.sim_thread.start()
            logger.info("Starting camera thread")
            sim.cam_thread.start()

            sim.sim_thread.join()

    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    init_eval_logging()
    logger.info("Holosoma Direct Simulation Runner (with depth cameras)")

    apply_runtime_patches()

    config = tyro.cli(
        RplRunSimConfig,
        description="Run simulation with depth cameras and image server.\n\n"
        "Usage: python -m holosoma_rpl.run_sim robot:<robot> terrain:<terrain>\n"
        "Examples:\n"
        "  python -m holosoma_rpl.run_sim robot:g1-29dof "
        "terrain:terrain_locomotion_stairs_and_slope_eval",
        config=TYRO_CONIFG,
    )

    run_simulation(config)


if __name__ == "__main__":
    main()
