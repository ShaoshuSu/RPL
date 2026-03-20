# holosoma-extension-rpl

Private extension for [holosoma](https://github.com/amazon-far/holosoma) adding depth camera sim2sim and sim2real support for the G1 robot with loco-manip policy.

## Setup

```bash
git clone --recurse-submodules <this-repo-url>
cd holosoma-extension-rpl

# Simulator environment (hsmujoco-rpl)
bash scripts/setup_mujoco.sh            # with GPU
bash scripts/setup_mujoco.sh --no-warp  # CPU-only

# Inference environment (hsinference-rpl)
bash scripts/setup_inference.sh
```

## Sim2sim

**Terminal 1 — Simulator:**
```bash
source scripts/source_mujoco_setup.sh
python -m holosoma_rpl.run_sim robot:g1-29dof terrain:terrain_locomotion_stairs--simulator.config.bridge.enabled=True
```

**Terminal 2 — Policy:**
```bash
source scripts/source_inference_setup.sh
python -m holosoma_rpl_inference.run_policy inference:g1-29dof-loco-manip-stand-height-waist
```

## Working with the Submodule

This repo uses `holosoma` as a git submodule under `thirdparty/holosoma`.

**First time setup** (if you cloned without `--recurse-submodules`):
```bash
git submodule update --init --recursive
```

**Pull latest submodule changes from remote:**
```bash
git submodule update --remote thirdparty/holosoma
git add thirdparty/holosoma
git commit -m "Update holosoma submodule to latest"
```

**Make changes to the submodule and submit a PR:**
```bash
cd thirdparty/holosoma
git checkout -b my-feature
# make changes, commit them
git push origin my-feature
# → open PR on github.com/amazon-far/holosoma
```

After the PR is merged, update the parent repo to point to the new commit:
```bash
cd ../..
git add thirdparty/holosoma
git commit -m "Update holosoma submodule after my-feature merged"
git push
```

> **Note:** The parent repo stores a pointer to a specific submodule commit. Any time the submodule moves to a new commit, you need to `git add` + `git commit` in the parent repo to update that pointer.

## Architecture

This repo uses `holosoma` as a git submodule under `thirdparty/` and extends it with two pip-installable packages:

- **`holosoma_rpl`** — Simulator extension: depth camera injection into MuJoCo, image server (shared memory streaming), extended terrain configs
- **`holosoma_rpl_inference`** — Inference extension: `LocoManipStandHeightWaitDepthPolicy`, camera/observation/task configs, ONNX model

Config classes inherit from holosoma's base configs (`RplRunSimConfig` extends `RunSimConfig`, `RplInferenceConfig` extends `InferenceConfig`). Runtime behavior (camera injection, image server threading) is applied via lightweight patches in `patches.py`.
