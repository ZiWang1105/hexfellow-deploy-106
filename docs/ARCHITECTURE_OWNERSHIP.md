# Dependency direction

The robot runtime is the reusable package. It does not import its consuming
agent application. `workspace.activate()` selects a configuration directory only;
it no longer checks for application source or adds that source to `sys.path`.

The default checkout contains all configurations and assets needed by the IK
viewer. The PinkIK/native implementation, process manager and shared-memory topics are
owned source modules in this repository. Robot URDF, meshes and configuration
live under `robot_resources`. There are no solver or process-runtime submodules.
Original attribution and source hashes are retained in NOTICE and the provenance manifest.
The robot adapter, runtime and application use one canonical controller instance
and command contract, not separate copies of the IK implementation.

Optional applications install and select their own component/plugin. The former
agent component and agent-specific tests moved to the application. Compatibility
imports in the application delegate to this package. Task prompts, model API
clients, grounding and offline task scoring remain outside the robot runtime.

Changing repository ownership does not identify physical actuator dynamics or
validate new motion. Standalone simulator evidence is recorded separately from
hardware qualification.
