"""Export an RPL terrain teacher (privileged height-map expert) to an ONNX that sim2sim can run.

The teachers in rpl_training/data/models/<terrain>/ are PPO-MA checkpoints with two actors:
- lower_body (15 actions: 12 legs + 3 waist): CNN on the 16x10 height map (conv 4x4/2 -> ReLU -> conv 4x4/2,
  16 channels each, 128 features) concatenated before the 575-dim state history, then MLP 512-256-256-128 (ELU);
- upper_body (14 arm actions): MLP 512-256-256-128 (ELU) on the 575-dim state history.

The ONNX keeps the depth student's interface so the sim2sim inference code can run it:
    actor_obs_lower_body [1, 575 + 160] = [actor_state_obs history | height map]
    actor_obs_upper_body [1, 575]       = actor_state_obs history
    action               [1, 29]        = [lower_body mean | upper_body mean] in motor order
The height map is pelvis_z - 0.7 - terrain_z on a yaw-aligned grid x = linspace(-0.8, 0.8, 16),
y = linspace(-0.5, 0.5, 10) (x-major), clipped to [-0.5, 0.5]; see rpl_sim2sim/height_scan.py.

Needs torch and onnx (the sim2sim conda envs have neither); runs on CPU:

    python rpl_sim2sim/tools/export_teacher_onnx.py --terrain stairs_up
    python rpl_sim2sim/tools/export_teacher_onnx.py --all

Before exporting, the rebuilt network is compared against holosoma's own PPOActorEncoder / PPOActor
(built from the checkpoint's experiment_config) on random inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
TEACHERS = ROOT / "rpl_training" / "data" / "models"
MODELS = ROOT / "rpl_sim2sim" / "models"
STUDENT_ONNX = MODELS / "rpl_holosoma_model_19999.onnx"
TERRAINS = ["stairs_up", "stairs_down", "slopes", "stepping_stones_columns"]
STATE_DIM = 575
SCAN_X, SCAN_Y = 16, 10
# Metadata copied from the student export: the inference code reads joint names / gains from it.
STUDENT_META_KEYS = ["dof_names", "kp", "kd", "command_ranges", "robot_urdf", "robot_urdf_path", "body_keys"]


def mlp(sd: dict, prefix: str) -> nn.Sequential:
    layers = []
    idx = sorted({int(k.split(".")[-2]) for k in sd if k.startswith(prefix) and k.endswith(".weight")})
    for n, i in enumerate(idx):
        w = sd[f"{prefix}{i}.weight"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(sd[f"{prefix}{i}.bias"])
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class TeacherActors(nn.Module):
    def __init__(self, lower_sd: dict, upper_sd: dict):
        super().__init__()
        c0, c2 = lower_sd["actor_module.encoder.cnn_layers.0.weight"], lower_sd["actor_module.encoder.cnn_layers.2.weight"]
        conv0 = nn.Conv2d(c0.shape[1], c0.shape[0], kernel_size=4, stride=2, padding=1)
        conv2 = nn.Conv2d(c2.shape[1], c2.shape[0], kernel_size=4, stride=2, padding=1)
        for conv, key in ((conv0, "0"), (conv2, "2")):
            conv.weight.data.copy_(lower_sd[f"actor_module.encoder.cnn_layers.{key}.weight"])
            conv.bias.data.copy_(lower_sd[f"actor_module.encoder.cnn_layers.{key}.bias"])
        # holosoma build_cnn_layer: activation after every conv except the last
        self.encoder = nn.Sequential(conv0, nn.ReLU(), conv2)
        self.lower = mlp(lower_sd, "actor_module.module.")
        self.upper = mlp(upper_sd, "actor_module.module.")

    def forward(self, actor_obs_lower_body: torch.Tensor, actor_obs_upper_body: torch.Tensor) -> torch.Tensor:
        state = actor_obs_lower_body[:, :STATE_DIM]
        scan = actor_obs_lower_body[:, STATE_DIM:].reshape(-1, 1, SCAN_X, SCAN_Y)
        features = self.encoder(scan).flatten(1)
        lower = self.lower(torch.cat([features, state], dim=1))
        upper = self.upper(actor_obs_upper_body)
        return torch.cat([lower, upper], dim=1)


def holosoma_reference(ckpt: dict):
    """holosoma's own actor classes, built from the checkpoint config, for verification."""
    sys.path.insert(0, str(ROOT / "thirdparty" / "holosoma" / "src" / "holosoma"))
    from holosoma.agents.modules.ppo_modules import PPOActor, PPOActorEncoder
    from holosoma.config_types.algo import LayerConfig, ModuleConfig

    cfg = ckpt["experiment_config"]["algo"]["config"]["module_dict"]
    obs_dims = {"actor_state_obs": STATE_DIM, "perception_obs": SCAN_X * SCAN_Y}
    history = {"actor_state_obs": 5, "perception_obs": 1}
    actors = {}
    for body, cls, n in (("lower_body", PPOActorEncoder, 15), ("upper_body", PPOActor, 14)):
        mc = cfg[f"actor_{body}"]
        module_cfg = ModuleConfig(
            type=mc["type"], input_dim=list(mc["input_dim"]), output_dim=list(mc["output_dim"]),
            layer_config=LayerConfig(**mc["layer_config"]),
        )
        actor = cls(obs_dims, module_cfg, n, 1.0, history)
        actor.load_state_dict(ckpt["actor_model_state_dict"][body])
        actors[body] = actor.eval()
    return actors


def export(terrain: str, student_meta: dict) -> Path:
    run_dir = next((TEACHERS / terrain).iterdir())
    ckpt_path = sorted(run_dir.glob("model_*.pt"))[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["actor_model_state_dict"]
    model = TeacherActors(sd["lower_body"], sd["upper_body"]).eval()

    ref = holosoma_reference(ckpt)
    torch.manual_seed(0)
    lower_in = torch.randn(64, STATE_DIM + SCAN_X * SCAN_Y)
    upper_in = lower_in[:, :STATE_DIM].clone()
    with torch.no_grad():
        ours = model(lower_in, upper_in)
        theirs = torch.cat([
            ref["lower_body"].act_inference({"actor_obs": lower_in}),
            ref["upper_body"].act_inference({"actor_obs": upper_in}),
        ], dim=1)
    err = (ours - theirs).abs().max().item()
    print(f"[{terrain}] {ckpt_path.relative_to(ROOT)}: max |rebuilt - holosoma| = {err:.2e}")
    if err > 1e-5:
        raise SystemExit(f"[{terrain}] rebuilt network does not match holosoma's actor (max err {err})")

    out = MODELS / f"rpl_teacher_{terrain}_{ckpt_path.stem}.onnx"
    torch.onnx.export(
        model, (lower_in[:1], upper_in[:1]), str(out), dynamo=False, opset_version=17,
        input_names=["actor_obs_lower_body", "actor_obs_upper_body"], output_names=["action"],
    )
    m = onnx.load(str(out))
    # holosoma_inference json-decodes every metadata value, so each one is stored JSON-encoded
    meta = {k: student_meta[k] for k in STUDENT_META_KEYS}
    meta.update({
        "policy": json.dumps("rpl_heightmap_teacher"),
        "teacher_terrain": json.dumps(terrain),
        "teacher_checkpoint": json.dumps(str(ckpt_path.relative_to(ROOT))),
        "iteration": json.dumps(int(ckpt["iteration"])),
        "wandb_run_path": json.dumps(str(ckpt["wandb_run_path"])),
        "experiment_config": json.dumps(ckpt["experiment_config"], default=repr),
    })
    for k, v in meta.items():
        json.loads(v)
    del m.metadata_props[:]
    for k, v in meta.items():
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, str(out))

    # batch size is fixed at 1, like the student export
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    err_onnx = max(
        abs(sess.run(None, {"actor_obs_lower_body": lower_in[i : i + 1].numpy(),
                            "actor_obs_upper_body": upper_in[i : i + 1].numpy()})[0] - theirs[i : i + 1].numpy()).max()
        for i in range(8)
    )
    print(f"[{terrain}] wrote {out.relative_to(ROOT)}; max |onnx - holosoma| = {err_onnx:.2e}")
    if err_onnx > 1e-4:
        raise SystemExit(f"[{terrain}] ONNX output does not match holosoma's actor (max err {err_onnx})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--terrain", choices=TERRAINS)
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()
    student_meta = {p.key: p.value for p in onnx.load(str(STUDENT_ONNX)).metadata_props}
    for terrain in TERRAINS if args.all else [args.terrain]:
        export(terrain, student_meta)


if __name__ == "__main__":
    main()
