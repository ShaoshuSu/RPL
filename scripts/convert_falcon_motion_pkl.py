"""Convert a FAR-FALCON-Terrains motion pkl (PHC format) into Holosoma ``MotionLib`` npz clips.

Only the joint-space data is meaningful: loco-manip upper-body reference commands read
``dof_pos`` / ``dof_vel`` alone, so the root pose comes from the pkl and the body-level
fields are single-body placeholders. ``dof_vel`` reproduces FALCON's forward difference
at the clip's native fps (``torch_humanoid_batch.fk_batch``), including its
repeated second-to-last frame.

Usage:
    python scripts/convert_falcon_motion_pkl.py \
        ../FAR-FALCON-Terrains/humanoidverse/data/motions/g1_29dof/v1/accad_all.pkl \
        src/holosoma/holosoma/data/motions/g1/falcon_accad_all
"""

import argparse
import re
from pathlib import Path

import joblib
import numpy as np


def convert(pkl_path: Path, out_dir: Path) -> int:
    motions = joblib.load(pkl_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, m) in enumerate(motions.items()):
        dof = np.asarray(m["dof"], dtype=np.float32)
        fps = int(m["fps"])
        num_frames = dof.shape[0]
        dof_vel = (dof[1:] - dof[:-1]) * fps
        dof_vel = np.concatenate([dof_vel, dof_vel[-2:-1]], axis=0)
        root_pos = np.asarray(m["root_trans_offset"], dtype=np.float32)
        root_quat = np.asarray(m["root_rot"], dtype=np.float32)  # placeholder; unused by loco-manip
        joint_pos = np.concatenate([root_pos, root_quat, dof], axis=1)
        joint_vel = np.concatenate([np.zeros((num_frames, 6), dtype=np.float32), dof_vel], axis=1)
        zeros3 = np.zeros((num_frames, 1, 3), dtype=np.float32)
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_frames, 1, 1))
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        np.savez_compressed(
            out_dir / f"{i:03d}_{safe}.npz",
            fps=np.array([fps]),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=zeros3,
            body_quat_w=quat,
            body_lin_vel_w=zeros3,
            body_ang_vel_w=zeros3,
        )
    return len(motions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pkl", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    print(f"converted {convert(args.pkl, args.out_dir)} motions -> {args.out_dir}")
