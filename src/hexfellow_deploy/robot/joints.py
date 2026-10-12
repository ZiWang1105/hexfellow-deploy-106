"""Physical encoder channels, separate from simulator-only chassis coordinates."""

BASE = "maver_x4_asm_maver_x4_base_link"
LIFT = "maver_x4_asm_URDF_joint_1"
LEFT = tuple(f"maver_x4_asm_7_urdf_SLDASM_joint_{i}" for i in range(1, 8))
RIGHT = tuple(f"maver_x4_asm_7_urdf_SLDASM2_joint_{i}" for i in range(1, 8))
JOINTS = (LIFT, *LEFT, *RIGHT)
SIDES = ("left", "right")
