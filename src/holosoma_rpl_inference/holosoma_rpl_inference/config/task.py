"""Task config for loco-manip stand-height-waist policy."""

from pathlib import Path

from holosoma_inference.config.config_types.task import TaskConfig

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_DEFAULT_MODEL_PATH = str(_MODELS_DIR / "loco_manip" / "model_19999.onnx")

loco_manip_stand_height_waist = TaskConfig(
    model_path=_DEFAULT_MODEL_PATH,
    rl_rate=50,
    policy_action_scale=0.25,
    use_phase=True,
    gait_period=1.0,
    desired_base_height=0.75,
    residual_upper_body_action=False,
    interface="lo",
    use_joystick=False,
    joystick_type="xbox",
    joystick_device=0,
    use_ros=False,
    wandb_download_dir="/tmp",
)
