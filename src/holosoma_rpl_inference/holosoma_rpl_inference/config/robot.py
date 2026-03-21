"""Robot config variant for loco-manip stand-height-waist policy."""

from dataclasses import replace

from holosoma_inference.config.config_values.robot import g1_29dof

# fmt: off
MOTOR_KP = [
    100, 100, 100, 150, 40, 40,
    100, 100, 100, 150, 40, 40,
    200, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
]

MOTOR_KD = [
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
    5.0, 5.0, 5.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
]

DEFAULT_DOF_ANGLES = [
    -0.1,   # left_hip_pitch_joint
    0.0,    # left_hip_roll_joint
    0.0,    # left_hip_yaw_joint
    0.3,    # left_knee_joint
    -0.2,   # left_ankle_pitch_joint
    0.0,    # left_ankle_roll_joint
    -0.1,   # right_hip_pitch_joint
    0.0,    # right_hip_roll_joint
    0.0,    # right_hip_yaw_joint
    0.3,    # right_knee_joint
    -0.2,   # right_ankle_pitch_joint
    0.0,    # right_ankle_roll_joint
    0.0,    # waist_yaw_joint
    0.0,    # waist_roll_joint
    0.0,    # waist_pitch_joint
    0.0,    # left_shoulder_pitch_joint
    0.0,    # left_shoulder_roll_joint
    0.0,    # left_shoulder_yaw_joint
    0.0,    # left_elbow_joint
    0.0,    # left_wrist_roll_joint
    0.0,    # left_wrist_pitch_joint
    0.0,    # left_wrist_yaw_joint
    0.0,    # right_shoulder_pitch_joint
    0.0,    # right_shoulder_roll_joint
    0.0,    # right_shoulder_yaw_joint
    0.0,    # right_elbow_joint
    0.0,    # right_wrist_roll_joint
    0.0,    # right_wrist_pitch_joint
    0.0,    # right_wrist_yaw_joint
]
# fmt: on

g1_29dof_loco_manip_stand_height_waist = replace(
    g1_29dof,
    default_dof_angles=DEFAULT_DOF_ANGLES,
    default_motor_angles=DEFAULT_DOF_ANGLES,
    motor_kp=MOTOR_KP,
    motor_kd=MOTOR_KD,
)
