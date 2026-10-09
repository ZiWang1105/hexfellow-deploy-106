# D405 left fixed-32 planning assets

Copied byte-for-byte from `/home/ziwang/artifacts/d405_left_wrist_v1` on 2026-10-08.
CAD source: `/home/ziwang/projects/PoolBot/cad/d405_left_wrist/build.py`.
`design_and_validation.json` records the original build checks and provenance;
those checks are not a new physical fit or structural qualification.

`assembly_mm/01_D405_LEFT_fixed32_wrist_mount.stl` is in millimeters and in its
assembled CAD frame (not the print orientation). `reference/` includes the
official D405 housing mesh in the rear mounting-plane camera frame and its
source/provenance record. The housing mesh is a camera reference, not a part to
print. The official ROS mesh source recorded in that JSON is the RealSense
ROS repository (Apache-2.0); original `_d405.urdf.xacro` documentation establishes
the nominal camera-frame relationship.

Actual installation is selected separately in
`config/camera/left_fixed32_installation.json`. Nominal CAD optical location is
not a measured calibration. The generated left pose plan hashes every asset.
