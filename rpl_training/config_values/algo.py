from dataclasses import replace

from holosoma.config_types.algo import (
    BodyPartBoolConfig,
    BodyPartFloatConfig,
    BodyPartLossTypeConfig,
    DAggerMAModuleDictConfig,
    DAggerMAMoEAlgoConfig,
    DAggerMAMoEConfig,
    LayerConfig,
    ModuleConfig,
    OptimizerConfig,
    PPOMAAlgoConfig,
    PPOMAConfig,
    PPOMultiModuleDictConfig,
    WarmupParams,
)

ppo_ma = PPOMAAlgoConfig(
    _target_="holosoma.agents.ppo.ppo_ma.PPOMultiActorCritic",
    _recursive_=False,
    config=PPOMAConfig(
        log_raw_episode=False,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        actor_learning_rate=1e-5,
        actor_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.01),
        critic_learning_rate=1e-5,
        critic_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.01),
        max_grad_norm=1.0,
        schedule="adaptive",
        desired_kl=0.01,
        use_symmetry=True,
        symmetry_actor_coef=1.0,
        symmetry_critic_coef=0.0,
        num_steps_per_env=24,
        save_interval=100,
        load_optimizer=True,
        init_noise_std_lower_body=0.8,
        init_noise_std_upper_body=0.6,
        num_learning_iterations=50000,
        init_at_random_ep_len=True,
        eval_callbacks=None,
        module_dict=PPOMultiModuleDictConfig(
            actor_lower_body=ModuleConfig(
                type="MLP",
                input_dim=["actor_state_obs", "perception_obs"],
                output_dim=["robot_action_dim"],
                layer_config=LayerConfig(hidden_dims=[512, 256, 256, 128], activation="ELU"),
            ),
            critic_lower_body=ModuleConfig(
                type="MLP",
                input_dim=["critic_state_obs", "perception_obs_wo_noise"],
                output_dim=[1],
                layer_config=LayerConfig(hidden_dims=[512, 256, 256, 128], activation="ELU"),
            ),
            actor_upper_body=ModuleConfig(
                type="MLP",
                input_dim=["actor_state_obs"],
                output_dim=["robot_action_dim"],
                layer_config=LayerConfig(hidden_dims=[512, 256, 256, 128], activation="ELU"),
            ),
            critic_upper_body=ModuleConfig(
                type="MLP",
                input_dim=["critic_state_obs"],
                output_dim=[1],
                layer_config=LayerConfig(hidden_dims=[512, 256, 256, 128], activation="ELU"),
            ),
        ),
    ),
)

# Shared CNN geometry for height-map encoders.  Ported from FAR-FALCON-Terrains
# ``config/algo/dagger_ma_moe_hm_cnn_cnn.yaml``.  ``input_height`` / ``input_width``
# must match the terrain scan grid (``height_scan_points_x_num`` /
# ``height_scan_points_y_num`` on :class:`TerrainTermCfg`), whose defaults are
# 16 / 10 (FAR-FALCON ``measured_points_x_num=16, measured_points_y_num=10``).
# ``perception_obs`` / ``perception_obs_wo_noise`` have ``history_length=1`` so
# ``input_channels=1``.
_CNN_HM_LAYER = LayerConfig(
    hidden_dims=[512, 256, 256, 128],
    activation="ELU",
    encoder_activation="ReLU",
    input_channels=1,
    input_height=16,
    input_width=10,
    hidden_channels=(16, 16),
    kernel_size=4,
    stride=2,
    padding=1,
    dropout_prob=0.0,
    use_layer_norm=False,
)

# PPO-MA with a CNN encoder on the height-map perception obs for the lower body.
# Upper body is MLP and does NOT consume ``perception_obs`` (matches FAR-FALCON's
# ``ppo_ma_hm.yaml``: ``actor_upper_body.input_dim=[actor_state_obs]``,
# ``critic_upper_body.input_dim=[critic_state_obs]``). Hidden dims
# ``[512, 256, 256, 128]`` also mirror FALCON's ``ppo_ma.yaml`` base.
ppo_ma_cnn = replace(
    ppo_ma,
    config=replace(
        ppo_ma.config,
        module_dict=replace(
            ppo_ma.config.module_dict,
            actor_lower_body=ModuleConfig(
                type="CNNEncoder",
                input_dim=["actor_state_obs", "perception_obs"],
                output_dim=["robot_action_dim"],
                layer_config=replace(
                    _CNN_HM_LAYER,
                    encoder_input_name="perception_obs",
                    module_input_name=("actor_state_obs",),
                ),
            ),
            critic_lower_body=ModuleConfig(
                type="CNNEncoder",
                input_dim=["critic_state_obs", "perception_obs_wo_noise"],
                output_dim=[1],
                layer_config=replace(
                    _CNN_HM_LAYER,
                    encoder_input_name="perception_obs_wo_noise",
                    module_input_name=("critic_state_obs",),
                ),
            ),
        ),
    ),
)


dagger_ma_moe = DAggerMAMoEAlgoConfig(
    _target_="holosoma.agents.dagger.dagger_ma_moe.DAggerMultiActorMoE",
    _recursive_=False,
    config=DAggerMAMoEConfig(
        log_raw_episode=False,
        num_learning_iterations=5000,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=24,
        save_interval=100,
        init_at_random_ep_len=True,
        actor_learning_rate=1e-4,
        actor_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.01),
        max_grad_norm=1.0,
        bc_loss_coef=1.0,
        init_noise_std_lower_body=0.8,
        init_noise_std_upper_body=0.6,
        loss_types=BodyPartLossTypeConfig(lower_body="mse", upper_body="mse"),
        use_encoder_loss=BodyPartBoolConfig(lower_body=False, upper_body=False),
        encoder_loss_coef=BodyPartFloatConfig(lower_body=1.0, upper_body=0.0),
        use_critic=False,
        use_scheduler=True,
        scheduler_type="CosineAnnealingLR",
        num_scheduler_iterations=5000,
        scheduler_params={"eta_min": 1e-6},
        use_warmup=True,
        warmup_params=WarmupParams(warmup_steps=100, warmup_start_factor=1e-3, warmup_end_factor=1.0),
        load_optimizer=True,
        eval_callbacks=None,
        route_obs_name="terrain_idx_obs",
        module_dict=DAggerMAModuleDictConfig(
            actor_lower_body=ModuleConfig(
                type="MLP",
                input_dim=["actor_state_obs", "perception_obs"],
                output_dim=["robot_action_dim"],
                layer_config=LayerConfig(hidden_dims=[512, 256, 256, 128], activation="ELU"),
            ),
            actor_upper_body=ModuleConfig(
                type="MLP",
                input_dim=["actor_state_obs"],
                output_dim=["robot_action_dim"],
                layer_config=LayerConfig(hidden_dims=[512, 256, 256, 128], activation="ELU"),
            ),
            actor_teacher_lower_body=ModuleConfig(
                type="MoEMLP",
                input_dim=["actor_state_obs", "perception_obs_wo_noise"],
                output_dim=["robot_action_dim"],
                layer_config=LayerConfig(
                    hidden_dims=[512, 256, 256, 128],
                    activation="ELU",
                    # Populate at experiment time with per-terrain proportions / names.
                    terrain_proportions=[],
                    terrain_types=[],
                ),
            ),
            actor_teacher_upper_body=ModuleConfig(
                type="MoEMLP",
                input_dim=["actor_state_obs"],
                output_dim=["robot_action_dim"],
                layer_config=LayerConfig(
                    hidden_dims=[512, 256, 256, 128],
                    activation="ELU",
                    terrain_proportions=[],
                    terrain_types=[],
                ),
            ),
            critic_lower_body=ModuleConfig(
                type="MLP",
                input_dim=["critic_state_obs", "perception_obs_wo_noise"],
                output_dim=[1],
                layer_config=LayerConfig(hidden_dims=[512, 256, 256, 128], activation="ELU"),
            ),
            critic_upper_body=ModuleConfig(
                type="MLP",
                input_dim=["critic_state_obs"],
                output_dim=[1],
                layer_config=LayerConfig(hidden_dims=[512, 256, 256, 128], activation="ELU"),
            ),
        ),
    ),
)


# Decoupled DAgger + MoE teacher with depth-camera VisionTransformer student.
# Ports FAR-FALCON-Terrains ``config/algo/dagger_ma_moe_hm_cnn_depth_vtr.yaml``:
#   * lower-body student = VisionTransformerEncoder over depth-camera ``perception_obs``
#                          (CNN tokeniser + cross/self attention + MLP head)
#   * lower-body teacher = MoE CNN encoder over height-map ``perception_obs_wo_noise``
#                          (per-terrain experts)
#   * lower-body encoder teacher = plain CNN encoder over the height map
#                                  (for optional encoder-distillation loss)
#   * upper body         = plain MLP student, MoE MLP teacher (no perception input)
#
# Vision-mask, attention/aggregation knobs and history lengths follow the FALCON YAML.
# The student encoder dimensions assume that ``perception_obs`` is a (resized) depth
# image stack; defaults below match the dual-depth resize (48x27, see
# ``config_values/loco/g1/camera.py``) but can be overridden on the CLI.
_VTR_DEPTH_LAYER_LOWER = LayerConfig(
    hidden_dims=[512, 256, 256, 128],
    activation="ELU",
    encoder_activation="ReLU",
    # VisionTransformer
    embed_dim=64,
    num_heads=4,
    num_blocks=4,
    attention_type="self",
    aggregation_method="last",
    use_pos_encoding=True,
    use_ffn=True,
    use_final_ln=False,
    ffn_hidden_dim=256,
    dropout_prob=0.0,
    use_layer_norm=False,
    # Vision mask gated by the standalone linear-velocity command observation group.
    use_vision_mask=True,
    vision_mask_input_name="command_lin_vel_obs",
    vision_mask_k=4.0,
    vision_mask_v_th=0.2,
    vision_mask_hard_cutoff=0.1,
    # CNN tokeniser geometry. Two cameras, depth resized to 48x27 by default.
    num_cameras=2,
    input_channels=2,
    input_height=27,
    input_width=48,
    hidden_channels=(16, 32, 32),
    kernel_size=4,
    stride=2,
    padding=1,
)

# CNN encoder geometry used by the MoE teacher and the (optional) encoder teacher,
# operating on the height-map ``perception_obs_wo_noise`` grid.
_HM_CNN_LAYER_TEACHER = LayerConfig(
    hidden_dims=[512, 256, 256, 128],
    activation="ELU",
    encoder_activation="ReLU",
    input_channels=1,
    input_height=16,
    input_width=10,
    hidden_channels=(16, 16),
    kernel_size=4,
    stride=2,
    padding=1,
    dropout_prob=0.0,
    use_layer_norm=False,
)

dagger_ma_moe_cnn_depth_vtr = replace(
    dagger_ma_moe,
    config=replace(
        dagger_ma_moe.config,
        # Match FALCON depth-vtr defaults: longer schedule, Adam, MSE loss.
        num_learning_iterations=20000,
        num_scheduler_iterations=10000,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        actor_optimizer=OptimizerConfig(_target_="torch.optim.Adam", weight_decay=0.0),
        critic_optimizer=OptimizerConfig(_target_="torch.optim.Adam", weight_decay=0.0),
        loss_types=BodyPartLossTypeConfig(lower_body="mse", upper_body="mse"),
        # Encoder-distillation on the lower body: the student VTR's CNN tokeniser
        # is matched against a CNN encoder teacher over the height map.
        use_encoder_loss=BodyPartBoolConfig(lower_body=False, upper_body=False),
        encoder_loss_coef=BodyPartFloatConfig(lower_body=1.0, upper_body=0.0),
        module_dict=replace(
            dagger_ma_moe.config.module_dict,
            actor_lower_body=ModuleConfig(
                type="VisionTransformerEncoder",
                input_dim=["actor_state_obs", "perception_obs", "command_lin_vel_obs"],
                output_dim=["robot_action_dim"],
                layer_config=replace(
                    _VTR_DEPTH_LAYER_LOWER,
                    encoder_input_name="perception_obs",
                    module_input_name=("actor_state_obs",),
                ),
            ),
            actor_teacher_lower_body=ModuleConfig(
                type="MoECNNEncoder",
                input_dim=["actor_teacher_state_obs", "perception_obs_wo_noise"],
                output_dim=["robot_action_dim"],
                layer_config=replace(
                    _HM_CNN_LAYER_TEACHER,
                    encoder_input_name="perception_obs_wo_noise",
                    module_input_name=("actor_teacher_state_obs",),
                    # Populate at experiment time with per-terrain proportions / names.
                    terrain_proportions=[],
                    terrain_types=[],
                ),
            ),
            actor_teacher_upper_body=ModuleConfig(
                type="MoEMLP",
                input_dim=["actor_teacher_state_obs"],
                output_dim=["robot_action_dim"],
                layer_config=LayerConfig(
                    hidden_dims=[512, 256, 256, 128],
                    activation="ELU",
                    terrain_proportions=[],
                    terrain_types=[],
                ),
            ),
            actor_encoder_teacher_lower_body=ModuleConfig(
                type="CNNEncoder",
                input_dim=["actor_state_obs", "perception_obs_wo_noise"],
                output_dim=["robot_action_dim"],
                layer_config=replace(
                    _HM_CNN_LAYER_TEACHER,
                    encoder_input_name="perception_obs_wo_noise",
                    module_input_name=("actor_state_obs",),
                ),
            ),
            # Upper body stays MLP / MoE MLP teacher (inherited from dagger_ma_moe).
        ),
    ),
)


# Exact FAR-FALCON-Terrains ``+exp=falcon_hm_dagger_moe`` student network. Same hyperparameters
# as :data:`dagger_ma_moe_cnn_depth_vtr` (checked against FALCON's resolved config), plus the
# two FALCON tokeniser details: channel tokens (32 tokens x 18 cells * 2 cameras, vision
# pos-encoding ``[1, 32, 64]``) and independently initialised per-camera CNN branches.
# Pair with ``observation:g1-29dof-falcon-hm-depth-distill-falcon`` (5-frame state history).
_VTR_DEPTH_LAYER_LOWER_FALCON = replace(
    _VTR_DEPTH_LAYER_LOWER,
    vision_tokens="channel",
    independent_camera_init=True,
)

dagger_ma_moe_cnn_depth_vtr_falcon = replace(
    dagger_ma_moe_cnn_depth_vtr,
    config=replace(
        dagger_ma_moe_cnn_depth_vtr.config,
        module_dict=replace(
            dagger_ma_moe_cnn_depth_vtr.config.module_dict,
            actor_lower_body=replace(
                dagger_ma_moe_cnn_depth_vtr.config.module_dict.actor_lower_body,
                layer_config=replace(
                    _VTR_DEPTH_LAYER_LOWER_FALCON,
                    encoder_input_name="perception_obs",
                    module_input_name=("actor_state_obs",),
                ),
            ),
        ),
    ),
)
