"""FAR-FALCON-Terrains parity checks for the depth-distillation networks.

* ``dagger_ma_moe_cnn_depth_vtr_falcon`` student uses FALCON's channel tokens.
* FALCON MoE CNN teacher checkpoints strict-load into the Holosoma teacher experts.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import holosoma
import pytest
import torch
from holosoma.agents.modules.module_utils import setup_student_teacher_actor_module
from holosoma.agents.modules.modules import BaseModule, build_cnn_layer

from rpl_training.config_values import algo as _algo_presets

DEFAULTS = vars(_algo_presets)

_FALCON_ALGO = DEFAULTS["dagger_ma_moe_cnn_depth_vtr_falcon"].config
_DEPTH_DIM = 2 * 2 * 27 * 48  # cameras * history * H * W
_OBS_DIMS = {
    "actor_state_obs": 115 * 5,
    "actor_teacher_state_obs": 115 * 5,
    "perception_obs": _DEPTH_DIM,
    "perception_obs_wo_noise": 160,
    "command_lin_vel_obs": 2,
}
_HISTORY = {
    "actor_state_obs": 5,
    "actor_teacher_state_obs": 5,
    "perception_obs": 2,
    "perception_obs_wo_noise": 1,
    "command_lin_vel_obs": 1,
}


def _student_module() -> BaseModule:
    cfg = replace(_FALCON_ALGO.module_dict.actor_lower_body, output_dim=[15])
    return BaseModule(_OBS_DIMS, cfg, {k: _HISTORY[k] for k in cfg.input_dim})


def test_falcon_vtr_uses_channel_tokens():
    vtr = _student_module().encoder
    # 32 CNN channels become tokens; each token holds 3x6 cells of both cameras.
    assert vtr.num_vision_tokens == 32
    assert vtr.vision_feat_dim == 36
    assert tuple(vtr.vision_pos_encoding.shape) == (1, 32, 64)
    out = vtr(torch.rand(3, _DEPTH_DIM), torch.randn(3, 575), v_cmd=torch.randn(3, 2))
    assert out.shape == (3, 64)


def test_default_vtr_keeps_spatial_tokens():
    cfg = DEFAULTS["dagger_ma_moe_cnn_depth_vtr"].config.module_dict.actor_lower_body
    vtr = BaseModule(
        {**_OBS_DIMS, "actor_state_obs": 115 * 16},
        replace(cfg, output_dim=[15]),
        {**_HISTORY, "actor_state_obs": 16},
    ).encoder
    assert (vtr.num_vision_tokens, vtr.vision_feat_dim) == (36, 32)


@pytest.mark.parametrize("independent", [False, True])
def test_independent_camera_init(independent: bool):
    layer = replace(_FALCON_ALGO.module_dict.actor_lower_body.layer_config, independent_camera_init=independent)
    torch.manual_seed(0)
    cnn = build_cnn_layer(
        layer.input_channels,
        layer.input_height,
        layer.input_width,
        layer.hidden_channels,
        layer.kernel_size,
        layer.stride,
        layer.padding,
        layer,
        flatten_output=False,
    )
    w0 = cnn.cnn_layers[0][0].weight
    w1 = cnn.cnn_layers[1][0].weight
    assert torch.equal(w0, w1) != independent
    assert w0.data_ptr() != w1.data_ptr()


def test_channel_tokens_vision_mask_zeroes_back_camera():
    cnn = _student_module().encoder.cnn_encoder
    feats = cnn(torch.rand(2, _DEPTH_DIM) + 0.1, v_cmd=torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    assert feats.shape == (2, 32, 36)
    front, back = feats[..., :18], feats[..., 18:]
    # Forward command -> back camera scale sigmoid-cut below hard_cutoff=0.1; reverse -> front.
    assert torch.all(back[0] == 0) and torch.any(front[0] != 0)
    assert torch.all(front[1] == 0) and torch.any(back[1] != 0)


def _teacher_ckpt() -> str | None:
    roots = [
        Path(holosoma.__file__).parent / "data/models/moe_cnn/falcon_20251105/stepping_stones_columns",
        Path(os.environ.get("FALCON_TERRAINS_ROOT", Path.home() / "lab42/src/FAR-FALCON-Terrains"))
        / "humanoidverse/data/models/moe_cnn/20251105/stepping_stone",
    ]
    for root in roots:
        found = sorted(str(f) for f in root.glob("*/model_*.pt"))
        if found:
            return found[-1]
    return None


@pytest.mark.parametrize(("body", "num_actions"), [("lower_body", 15), ("upper_body", 14)])
def test_falcon_teacher_checkpoint_strict_loads(body: str, num_actions: int):
    ckpt = _teacher_ckpt()
    if ckpt is None:
        pytest.skip("FALCON MoE teacher checkpoint not available")
    module_cfg = getattr(_FALCON_ALGO.module_dict, f"actor_teacher_{body}")
    module_cfg = replace(
        module_cfg,
        layer_config=replace(module_cfg.layer_config, terrain_proportions=[0.0, 0.0, 1.0], terrain_types=[]),
    )
    teacher = setup_student_teacher_actor_module(
        obs_dim_dict=_OBS_DIMS,
        module_config=module_cfg,
        num_actions=num_actions,
        init_noise_std=0.8,
        device="cpu",
        history_length={k: _HISTORY[k] for k in module_cfg.input_dim},
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_model_state_dict"][body]
    teacher.experts[0].load_state_dict(sd, strict=True)

    obs = torch.randn(4, sum(_OBS_DIMS[k] for k in module_cfg.input_dim))
    actions = teacher.act({"actor_obs": obs, "route_obs": torch.full((4, 1), 2.0)})
    assert actions.shape == (4, num_actions)
    assert torch.isfinite(actions).all()
