"""Read ideal physical-joint encoders, with no access to simulator root state.

The reader receives only the sensor array and clock, not MjData, qpos, qvel,
object handles or scene configuration. Base pose/velocity are not sensors in
this profile. Wheel encoders and IMU are absent, not fabricated from root state.
"""
from ..robot.joints import JOINTS, SIDES
from ..control.gripper import gripper_state


class EncoderReader:
    def __init__(self, model):
        def address(name):
            sensor = model.sensor(name)
            if int(sensor.dim[0]) != 1:
                raise ValueError("Encoder must be a scalar sensor")
            return int(sensor.adr[0])
        self.positions = {n: address("encoder_position_"+n) for n in JOINTS}
        self.velocities = {n: address("encoder_velocity_"+n) for n in JOINTS}
        self.jaws = {side: [address(f"encoder_position_{side}_jaw_{jaw}")
                           for jaw in ("positive", "negative")] for side in SIDES}

        self.open_width = {side: sum(float(model.joint(f"{side}_jaw_{jaw}").range[1])
                              for jaw in ("positive", "negative")) for side in SIDES}

    def read(self, sensor_values, timestamp_s):
        return {"timestamp_s": float(timestamp_s),
                "joint_positions": {n: float(sensor_values[i]) for n, i in self.positions.items()},
                "joint_velocities": {n: float(sensor_values[i]) for n, i in self.velocities.items()},
                "gripper_state": {side: gripper_state(sum(float(sensor_values[i]) for i in indices),
                                                     0., self.open_width[side])
                                  for side, indices in self.jaws.items()}}
