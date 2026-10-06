"""Recoverable per-pose failures, distinct from robot safety faults."""


class InvalidCalibrationPose(ValueError):
    """This target cannot provide a capture; stop/check HOLD before advancing."""

    def __init__(self, message, *, pose_id=None):
        super().__init__(message)
        self.pose_id = pose_id
