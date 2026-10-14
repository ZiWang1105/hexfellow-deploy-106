# ZED visualization assets

`zed2i.stl` uses metre coordinates, copied from PoolBot's camera model.
`zed_short_triangular_v5_15deg_mm.stl` is the exact short, ribbed v5 print artifact,
with millimetre coordinates and explicit `0.001` URDF mesh scale. Source:
`~/artifacts/poolbot_camera_mounts/zed_short_triangular_v5_15deg/`.

Shelf midplane: **65 mm**, 100 mm lower than old v3. Pitch: **15 degrees**.
Overall bracket bounds: 65 x 60 x 76.3094 mm. Camera mount transform is copied
from `proposed_camera_transform.json`; the robot-to-bracket attachment retains
the nominal PoolBot transform. Mount inertia is recomputed from this closed mesh
assuming uniform 1250 kg/m^3 density; this is a visualization model, not a verified
hardware payload. The robot hardware profiles are unchanged.

The earlier tall 165 mm / 32 degree STL was selected incorrectly. Those earlier
renders do not represent this shortened supporter. Input hashes and detailed
provenance are in the view-only robot model descriptor.

## Alternative tilt angles

`zed_short_triangular_{32,40,42,55}deg_mm.stl` keep the same 65 mm shelf datum,
base dimensions, mounting holes and triangular rib arrangement. Only shelf pitch
and the support's mating top edges change. Camera transforms and estimated inertias
match each STL in the corresponding `maver_x4_gp80_zed_*deg.urdf`. See
`cad/build_short_zed_variants.py` and `angle_variants_checks.json` for reproduction
and geometric checks. These prototypes have not been physically qualified.
