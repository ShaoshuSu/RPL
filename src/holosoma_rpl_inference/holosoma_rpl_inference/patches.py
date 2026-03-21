"""Runtime patches for holosoma_inference to support loco-manip depth policy."""

from loguru import logger


def _patch_select_policy_class(original_fn):
    """Wrap _select_policy_class to route configs with perception_obs to
    LocoManipStandHeightWaitDepthPolicy."""

    def patched(config):
        if "perception_obs" in config.observation.obs_dict:
            from holosoma_rpl_inference.policies.loco_manip_stand_height_wait_depth import (
                LocoManipStandHeightWaitDepthPolicy,
            )

            return LocoManipStandHeightWaitDepthPolicy
        return original_fn(config)

    return patched


def apply_runtime_patches():
    """Apply runtime-only patches to holosoma_inference."""
    from holosoma_inference.policies import dual_mode

    dual_mode._select_policy_class = _patch_select_policy_class(
        dual_mode._select_policy_class
    )

    logger.info("holosoma-rpl-inference runtime patches applied")
