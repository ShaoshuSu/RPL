"""RPL / FALCON-Terrains sim2sim.

Sim side: FAR-FALCON-Terrains' own MuJoCo sim2sim environment, vendored (``simulator``, ``bridge``,
``image_server``, ``rate_limiter``, ``assets/``, ``config/``; ``python -m rpl_sim2sim.run_sim``). It needs
MuJoCo, gymnasium and unitree_sdk2py, not holosoma. Policy side: the depth student for
``holosoma_inference`` (``policy``, ``inference_presets``; ``python -m rpl_sim2sim.run_policy``).
"""
