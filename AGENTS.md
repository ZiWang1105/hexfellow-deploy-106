# Hexfellow deployment instructions

Read `docs/STANDALONE.md`, `docs/ARCHITECTURE_OWNERSHIP.md`, and
`docs/GRIPPER_CONTROL.md` before changing the robot runtime.

- This repository must run the IK simulator and interactive viewer independently.
  Do not import `mobile_manipulator` or require an agent application checkout in
  robot code, configuration defaults, tests, or setup scripts. Applications are
  optional explicit plugins and depend on this package.
- Own robot control, measured sensor/action contracts, FK, hardware interfaces,
  simulator/assets, and gripper regulation here. Use the owned
  `hexfellow_deploy.robotics` solver and `hexfellow_deploy.runtime` process/topic
  implementation; preserve source attribution and asset hashes. Do not restore
  either external repository, package import, submodule, or setup dependency.
- Simulation and hardware share the controller and MIT reference/actuator contract.
  Keep measured-feedback, ownership, stop and command-validation requirements.
  The standalone example configuration is uncommissioned; no implicit hardware
  execution, automatic rearm, calibration promotion, or release-binding approval.
- The base/root truth and object/contact truth belong only to the simulator and
  offline evaluator. They must not enter robot-control or policy observations.
- Preserve existing user changes and calibration evidence. Do not copy research
  runs, API keys, or machine-local deployment state into standalone defaults.
- Run deployment tests in the deployment virtual environment. Native wire-format
  tests require the hardware extra but open no devices. Run calibration tests in
  a separate process with `MUJOCO_GL=egl` to keep calibration tool imports separate from runtime tests. Exercise standalone import rejection and shared simulation when
  changing dependency ownership. GUI changes also require an actual viewer test.
