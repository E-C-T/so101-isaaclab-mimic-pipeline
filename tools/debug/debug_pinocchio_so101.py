from __future__ import annotations

import argparse
import numpy as np


DEFAULT_URDF = (
    "/home/insol02/IH_ws/so101_IsaacLab/"
    "src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf"
)

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

ALL_JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]

# Episode-0 initial joint state from your annotation debug.
DEFAULT_Q = np.array(
    [-0.04838089, -1.6236473, 1.7326647, 1.3117621, 1.5566326, 0.03890861],
    dtype=np.float64,
)

# Isaac ee_frame pose from your annotation debug at episode-0 reset.
# This is the frame you currently call "end_effector".
DEFAULT_ISAAC_EEF_POS = np.array([0.11265052, 0.01338875, -0.02020334], dtype=np.float64)

# Your Isaac Lab FrameTransformer config uses:
# target frame = gripper_link
# offset = [0.01, 0.0, -0.09]
DEFAULT_EE_OFFSET = np.array([0.01, 0.0, -0.09], dtype=np.float64)


def import_pinocchio():
    try:
        import pinocchio as pin
    except Exception as exc:
        raise RuntimeError(
            "Could not import pinocchio. Try running this through the same env used for Isaac Lab, "
            "for example: cd /home/insol02/IH_ws/IsaacLab && "
            "./isaaclab.sh -p /home/insol02/IH_ws/so101_IsaacLab/tools/debug_pinocchio_so101.py"
        ) from exc
    return pin


def set_named_joint_q(model, q, joint_name, value):
    joint_id = model.getJointId(joint_name)
    if joint_id == 0 or joint_id >= len(model.joints):
        raise KeyError(f"Joint '{joint_name}' not found in Pinocchio model.")
    idx_q = model.joints[joint_id].idx_q
    q[idx_q] = value
    return idx_q


def get_frame_pose(pin, model, data, q, frame_name):
    frame_id = model.getFrameId(frame_name)
    if frame_id >= len(model.frames):
        raise KeyError(f"Frame '{frame_name}' not found in Pinocchio model.")
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF)
    parser.add_argument("--print-all-frames", action="store_true")
    parser.add_argument("--print-all-joints", action="store_true")
    parser.add_argument(
        "--q",
        type=float,
        nargs=6,
        default=DEFAULT_Q.tolist(),
        help="Joint vector: shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll gripper",
    )
    args = parser.parse_args()

    pin = import_pinocchio()

    print("[INFO] Loading URDF:")
    print("  ", args.urdf)
    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()

    print("[INFO] model.nq =", model.nq)
    print("[INFO] model.nv =", model.nv)

    if args.print_all_joints:
        print("\n[JOINTS]")
        for i, joint in enumerate(model.joints):
            print(
                f"{i:02d}: name={model.names[i]} "
                f"idx_q={joint.idx_q} nq={joint.nq} idx_v={joint.idx_v} nv={joint.nv}"
            )

    print("\n[REQUIRED JOINT INDEX CHECK]")
    q = pin.neutral(model)
    print("[INFO] neutral q =", q)

    q_input = np.array(args.q, dtype=np.float64)
    for name, value in zip(ALL_JOINT_NAMES, q_input):
        idx_q = set_named_joint_q(model, q, name, value)
        print(f"  {name:15s} -> q[{idx_q}] = {value:+.6f}")

    print("\n[INFO] q after assignment =", q)

    if args.print_all_frames:
        print("\n[FRAMES]")
        for i, frame in enumerate(model.frames):
            print(f"{i:03d}: {frame.name}")

    print("\n[GRIPPER FRAME SEARCH]")
    for i, frame in enumerate(model.frames):
        if "gripper" in frame.name or "wrist" in frame.name or "base" in frame.name:
            print(f"{i:03d}: {frame.name}")

    frame_names_to_check = ["gripper_link", "gripper_frame_link"]

    print("\n[FK RESULTS]")
    for frame_name in frame_names_to_check:
        try:
            T = get_frame_pose(pin, model, data, q, frame_name)
        except Exception as exc:
            print(f"[WARN] Could not compute frame '{frame_name}': {exc}")
            continue

        print(f"\nFrame: {frame_name}")
        print("translation =", T.translation)
        print("rotation =\n", T.rotation)

        # Compare plain frame origin to Isaac ee_frame position.
        pos_err = np.linalg.norm(T.translation - DEFAULT_ISAAC_EEF_POS)
        print("error_vs_isaac_eef_pos_no_offset =", pos_err)

        # Compare gripper_link + Isaac FrameTransformer offset.
        if frame_name == "gripper_link":
            T_offset = pin.SE3(np.eye(3), DEFAULT_EE_OFFSET)
            T_eef = T * T_offset
            print("\nFrame: gripper_link + Isaac offset [0.01, 0.0, -0.09]")
            print("translation =", T_eef.translation)
            print("rotation =\n", T_eef.rotation)
            pos_err_offset = np.linalg.norm(T_eef.translation - DEFAULT_ISAAC_EEF_POS)
            print("error_vs_isaac_eef_pos_with_offset =", pos_err_offset)

    print("\n[EXPECTED ISAAC DEBUG EEF POS]")
    print(DEFAULT_ISAAC_EEF_POS)

    print("\n[DONE]")
    print(
        "If gripper_link + offset or gripper_frame_link is close to the expected Isaac EEF position, "
        "we can use that as the IK target frame. If both are far, we need to inspect root-frame or "
        "URDF/USD convention mismatch before implementing IK."
    )


if __name__ == "__main__":
    main()
