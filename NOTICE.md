# Source attribution

The deployment owns its runtime and robotics implementations. They are edited,
built, tested, and versioned in this repository. Neither former package is
imported, installed, cloned, or resolved at runtime.

The initial robotics implementation was extracted from Wzz-D/UMI_Hardware at
`e4dbcc9983516e20d59363680c60066a8571c0e6`. No repository-wide license was found
at that revision; this extraction does not invent a license grant. Original
author comments and resource-specific notices remain.

The process/topic implementation derives from HaoruXue/deploy-anything at
`6c2bda40ebee56ab3a50b8dd62aaf4df5814e5a8`, under Apache-2.0. The complete
license is retained in `licenses/runtime-APACHE-2.0.txt`. This extraction changes
package ownership, import paths, build integration, resource resolution, and
source binding. It preserves the existing controller and communication behavior.

`docs/OWNED_RUNTIME_PROVENANCE.json` records exact source hashes and destination
paths. These historical names identify attribution, not live dependencies.
