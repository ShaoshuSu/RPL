"""Robot presets for RPL / FALCON-Terrains."""

from dataclasses import replace

from holosoma.config_values.robot import g1_29dof

# Mirrors FAR-FALCON-Terrains' ``g1_29dof_rev_1_0.yaml``.
# Overrides every attribute that diverged from the Holosoma defaults:
#   - dof_armature_list / dof_joint_friction_list
#   - dof_vel_limit_list (ankles 37 → 30)
#   - dof_effort_limit_list (already × FALCON's dof_effort_limit_scale=0.8; ankle 50 → 28, waist roll/pitch 50 → 28)
#   - init_state default joint angles (FALCON's stand pose)
#   - control.stiffness / control.damping (FALCON's Nm/rad PD gains)
#   - asset.armature (Holosoma 0.001 → 0.0; per-DOF armature covers it)
_G1_LEG_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
_G1_WAIST_DOF_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
_G1_LEG_PENALTY_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

g1_29dof_falcon = replace(
    g1_29dof,
    # Decoupled HM bodies as in FALCON: the lower-body agent owns legs + waist,
    # the upper-body agent owns the arms (Holosoma's ``g1_29dof`` puts the waist upstairs).
    lower_dof_names=_G1_LEG_DOF_NAMES + _G1_WAIST_DOF_NAMES,
    upper_dof_names=[n for n in g1_29dof.upper_dof_names if n not in _G1_WAIST_DOF_NAMES],
    # Lower-body penalty subsets used by the FALCON reward terms.
    lower_body_torque_penalty_dof_names=["left_knee_joint", "right_knee_joint"],
    lower_body_vel_penalty_dof_names=_G1_LEG_PENALTY_DOF_NAMES,
    lower_body_acc_penalty_dof_names=_G1_LEG_PENALTY_DOF_NAMES,
    # URDF / XML come from FAR-FALCON-Terrains (``g1_29dof_rev_1_0.*``). That
    # file hard-codes tighter ankle/waist effort and velocity limits (effort 35,
    # velocity 30 for ankles) than Holosoma's ``g1_29dof.urdf``. The lists below
    # mirror those URDF values exactly so the IsaacGym loader assertion passes.
    # FALCON's URDF does not include Holosoma's ``foot_contact_point`` auxiliary
    # frames, so override the feet-related names to ``ankle_roll_link`` (which
    # FALCON does expose), and drop the two ``foot_contact_point`` entries from
    # ``body_names`` so name resolution matches FALCON's topology.
    key_bodies=["left_ankle_roll_link", "right_ankle_roll_link"],
    foot_height_name="ankle_roll_link",
    # Matches FAR-FALCON's ``g1_29dof_rev_1_0.yaml: lidar_height_offset: 0.7``.
    # Used by the ``height_measurements`` obs term to bias the root-relative
    # depth signal sent to the CNN encoder.
    lidar_height_offset=0.7,
    body_names=[
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "left_rubber_hand",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
        "right_rubber_hand",
    ],
    dof_vel_limit_list=[
        32.0,
        20.0,
        32.0,
        20.0,
        30.0,
        30.0,  # left leg
        32.0,
        20.0,
        32.0,
        20.0,
        30.0,
        30.0,  # right leg
        32.0,
        30.0,
        30.0,  # waist yaw / roll / pitch
        37.0,
        37.0,
        37.0,
        37.0,  # left arm
        37.0,
        22.0,
        22.0,  # left wrist
        37.0,
        37.0,
        37.0,
        37.0,  # right arm
        37.0,
        22.0,
        22.0,  # right wrist
    ],
    dof_effort_limit_list=[
        88.0,
        139.0,
        88.0,
        139.0,
        35.0,
        35.0,  # left leg
        88.0,
        139.0,
        88.0,
        139.0,
        35.0,
        35.0,  # right leg
        88.0,
        35.0,
        35.0,  # waist
        25.0,
        25.0,
        25.0,
        25.0,  # left arm
        25.0,
        5.0,
        5.0,  # left wrist
        25.0,
        25.0,
        25.0,
        25.0,  # right arm
        25.0,
        5.0,
        5.0,  # right wrist
    ],
    dof_armature_list=[
        0.010177520,
        0.025101925,
        0.010177520,
        0.025101925,
        0.003609725,
        0.003609725,  # left leg
        0.010177520,
        0.025101925,
        0.010177520,
        0.025101925,
        0.003609725,
        0.003609725,  # right leg
        0.010177520,
        0.010177520,
        0.010177520,  # waist yaw / roll / pitch
        0.003609725,
        0.003609725,
        0.003609725,
        0.003609725,  # left arm
        0.003609725,
        0.003609725,
        0.003609725,  # left wrist
        0.003609725,
        0.003609725,
        0.003609725,
        0.003609725,  # right arm
        0.003609725,
        0.003609725,
        0.003609725,  # right wrist
    ],
    dof_joint_friction_list=[0.01] * 29,
    init_state=replace(
        g1_29dof.init_state,
        default_joint_angles={
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
    ),
    control=replace(
        g1_29dof.control,
        stiffness={
            "hip_yaw": 100.0,
            "hip_roll": 100.0,
            "hip_pitch": 100.0,
            "knee": 150.0,
            "ankle_pitch": 40.0,
            "ankle_roll": 40.0,
            "waist_yaw": 200.0,
            "waist_roll": 40.0,
            "waist_pitch": 40.0,
            "shoulder_pitch": 40.0,
            "shoulder_roll": 40.0,
            "shoulder_yaw": 40.0,
            "elbow": 40.0,
            "wrist_roll": 40.0,
            "wrist_pitch": 40.0,
            "wrist_yaw": 40.0,
        },
        damping={
            "hip_yaw": 2.0,
            "hip_roll": 2.0,
            "hip_pitch": 2.0,
            "knee": 4.0,
            "ankle_pitch": 2.0,
            "ankle_roll": 2.0,
            "waist_yaw": 5.0,
            "waist_roll": 5.0,
            "waist_pitch": 5.0,
            "shoulder_pitch": 1.0,
            "shoulder_roll": 1.0,
            "shoulder_yaw": 1.0,
            "elbow": 1.0,
            "wrist_roll": 1.0,
            "wrist_pitch": 1.0,
            "wrist_yaw": 1.0,
        },
    ),
    asset=replace(
        g1_29dof.asset,
        armature=0.0,
        urdf_file="g1/g1_29dof_rev_1_0.urdf",
        xml_file="g1/g1_29dof_rev_1_0.xml",
        robot_type="g1_29dof_rev_1_0",
    ),
)
