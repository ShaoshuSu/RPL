# RPL: FALCON-Terrains on Holosoma

Per-terrain RL experts and multi-depth DAgger distillation for the Unitree G1 (29 DoF),
reproducing FAR-FALCON-Terrains on top of [Holosoma](https://github.com/amazon-far/holosoma).

The repo follows the [php_parkour](https://github.com/amazon-far/php_parkour) layout:

| Path | Contents |
|---|---|
| `thirdparty/holosoma` | Holosoma, pinned to branch `zyuanhan/rpl-release-port` (core features RPL needs, on top of the PHP release port's warp depth sensor) |
| `rpl_training/` | Everything outside Holosoma core: loco-manipulation env, manager terms, presets, launchers, assets |
| `rpl_training/data/` | FALCON ACCAD upper-body clips and the four FALCON MoE teacher checkpoints (`moe_cnn/falcon_20251105`) |
| `rpl_sim2sim/` | Sim2sim: FAR-FALCON-Terrains' MuJoCo environment (sim side) and the depth student for `holosoma_inference` (policy side), plus two reference student exports |

What lives where:

- **Holosoma core (`zyuanhan/rpl-release-port`)**: warp ray-cast camera manager (delay buffer, D435i noise,
  pose / FOV randomization, robot self-occlusion), observation push intervals / history across resets /
  time-major history, multi-agent reward routing, PPO-MA, DAgger / DAgger-MA / DAgger-MA-MoE, MoE / ViT /
  channel-token CNN modules, FALCON terrains and heightfield lookups, terrain levels, foothold sampler,
  body-group control delay and DR changes, IsaacSim heightfield tiles.
- **`rpl_training`**: `envs/` (loco-manip env + end-effector forces), `terms/` (command, curriculum,
  observation, reward, termination), `config_values/` (FALCON presets), `training_runs/` (launchers), `tests/`.

## Setup

```bash
git clone --recurse-submodules https://github.com/amazon-far/RPL.git
cd RPL
SIM=isaacgym bash rpl_training/training_runs/setup_env.sh   # or SIM=isaacsim
```

`SIM=isaacgym` (FALCON's simulator, conda env `hsgym`) is the default; `SIM=isaacsim` uses `hssim`.
`ENV_NAME` overrides the env name and `HOLOSOMA_LOCAL_DIR` another holosoma checkout. Launchers log to
W&B by default (`LOGGER=disabled` keeps logs local under `logs/`).

## Train the RL experts

```bash
NGPUS=4 EXPERT=stones bash rpl_training/training_runs/run_expert.sh --training.name stones-expert
NGPUS=4 EXPERT=stairs bash rpl_training/training_runs/run_expert.sh --training.name stairs-expert
NGPUS=4 EXPERT=slopes bash rpl_training/training_runs/run_expert.sh --training.name slopes-expert
```

Presets: `exp:g1-29dof-falcon-hm-force-falcon-{stones,stairs,slopes}` reproduce the run args of
FALCON's final expert stages (`falcon-29-terrains-experts-stones/lrlo8qpt`,
`falcon-29-terrains-experts/h5d52hm0`, `falcon-29-terrains-experts/b04ael3p`): terrain mix, reward
weights / masks, friction, base COM, IMU and heightmap noise, stair tread width, rough slopes with
1 m tile spacing. They train from scratch with the curricula on and no lower-body control delay,
where FALCON fine-tuned from earlier stages. Their base,
`exp:g1-29dof-falcon-hm-force-falcon` (FALCON `+exp=falcon_hm_multi_terrain algo=ppo_ma_hm_cnn`),
takes the terrain mix on the CLI.

## Distill the multi-depth student

```bash
NGPUS=8 bash rpl_training/training_runs/run_distill.sh --training.name rpl-student
```

Preset: `exp:g1-29dof-falcon-hm-force-falcon-distill` (FALCON `+exp=falcon_hm_dagger_moe
algo=dagger_ma_moe_hm_cnn_depth_vtr`). It loads the shipped FALCON teachers by default; use
`--algo.config.ckpt-teacher DIR` (same per-terrain layout) or `--algo.config.ckpt-teacher-wandb.<terrain> wandb://...`
for your own experts.

`--training.num-envs` is the total across GPUs. Presets default to FALCON's totals (experts 4096 x 4 GPUs,
slopes expert 4096 total as in FALCON, student 1024 x 8 GPUs).

The presets are also registered as Holosoma entry points, so after `pip install -e .`
`python -m holosoma.train_agent exp:g1-29dof-falcon-hm-force-falcon ...` works too.

## Sim2sim (MuJoCo)

The sim is FAR-FALCON-Terrains' own sim2sim environment, vendored in `rpl_sim2sim`
(`sim2real/run_sim.py task=loco_manip/loco_manip_cam robot=g1/g1_29dof +obs=loco_manip/g1_29dof_depth
task.USE_JOYSTICK=true` with `scene_g1_29dof_terrains_freebase_real.xml`): its simulator loop (PD on the latest
`rt/lowcmd` every 5 ms, elastic band, passive viewer), Unitree `sdk2py` bridge, and depth image server (front +
back cameras, 240x135, gymnasium renderer, cv2 bicubic resize to 48x27, clip to [0.1, 2] m, normalized to
[-0.5, 0.5], `depth_delay` 1 frame, 10 Hz, shared memory `depth_img_shm`), with FALCON's MJCF and its resolved
config (`rpl_sim2sim/config/loco_manip_cam.yaml`). The policy side is the depth student in `holosoma_inference`.

```bash
bash rpl_sim2sim/setup_env.sh          # conda envs rplmujoco (sim) + rplinference (policy)
bash rpl_sim2sim/run_sim.sh            # terminal 1: FALCON course, Xbox pad on this host
bash rpl_sim2sim/run_policy.sh         # terminal 2: shipped student (or --task.model-path X.onnx)
```

Once both are up, release the band (`9` in the MuJoCo window) and start the policy right away.
Xbox pad (FALCON's mapping): `start` start, `A+X` init pose, `R2` stand/walk, left stick forward speed,
right stick yaw, `L2` zero velocity, `B+Y` stop, `B+up/down` base height, `Y`/`select` + d-pad waist.
For the keyboard instead, run `run_sim.sh --no-joystick` and add
`--task.velocity-input keyboard --task.state-input keyboard` to `run_policy.sh`; then `]` start,
`i` init pose, `=` walk, `w/s/a/d` velocity, `q/e` yaw, `o` stop. MuJoCo window: `7`/`8` band
length, `3`-`6` move its anchor, `R` toggles video recording (`logs/sim2sim/simulation.mp4`).
FALCON spawns the robot on the stair platform; `run_sim.sh --spawn 0 -2 0.8` starts it on the floor
facing the stairs. Distillation runs export `model_XXXXX.onnx` next to each checkpoint; pass one with
`--task.model-path`. `CONDA_ROOT=~/.falcon_deps/miniconda3 SIM_ENV=fcreal bash rpl_sim2sim/run_sim.sh`
runs the sim in FALCON's own env (MuJoCo 3.9).

## Tests

```bash
python -m pytest rpl_training/tests   # FALCON parity tests for terms, terrains, modules, warp camera
python -m pytest rpl_sim2sim/tests    # sim2sim: sim side in rplmujoco, policy side in rplinference
```
