# hand_v15 source reference

Pinned source: [wangyenjen/agentic-robotics-20260908](https://github.com/wangyenjen/agentic-robotics-20260908/tree/yenjen-old-code-dev/libraries/hand_v15), branch `yenjen-old-code-dev`, commit `facc274ff998d1545dc75f6ebe107e6da666dacd`, library version `3.11.0`.

`skills/` is a byte-for-byte reference snapshot of the requested library's Python files. `PROVENANCE.json` records their SHA-256 hashes and inventories all 38 enabled upstream entries, including entries supplied by upstream `physapi`. Those external `physapi` modules are hashed but not vendored. Integration reports, research logs, historical deployment state, robot assets and credentials are not copied.

These files are **not imported or executed**. They depend on upstream `physapi`, robot-specific kinematics, world-frame/table geometry, continuous jaw apertures and perception APIs that this workspace does not expose. The executable adaptation is `src/hexfellow_deploy/skills/`; see `docs/HAND_V15_SKILLS.md` for exact supported and unsupported behavior. No upstream success rates are evidence for this robot.

At the pinned commit, no root LICENSE file was present. Original source comments and author attribution are retained. This snapshot does not assert or grant a new license; resolve upstream licensing before redistributing it outside the authorized project.
