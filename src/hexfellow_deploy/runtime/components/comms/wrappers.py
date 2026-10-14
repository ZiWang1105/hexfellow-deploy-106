from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from typing_extensions import override

from hexfellow_deploy.runtime.components.comms.shm_comm import DASHMNumPyComm
from hexfellow_deploy.runtime.utils.data_types import Quat, Vec3


class ImuCommWrapper(DASHMNumPyComm):
    """Wrapper for the IMU comm."""

    @dataclass
    class ImuData:
        """Data class for the IMU comm."""

        timestamp: float
        linear_acceleration: Vec3
        angular_velocity: Vec3
        orientation: Quat

    def __init__(self, name: str, *, create: bool = True):
        super().__init__(name, (10,), np.float64, create=create)

    @override
    def send(self, data: ImuData, timeout: Optional[float] = None) -> bool:
        """Send the IMU data."""
        data_list = [
            data.linear_acceleration.x,
            data.linear_acceleration.y,
            data.linear_acceleration.z,
            data.angular_velocity.x,
            data.angular_velocity.y,
            data.angular_velocity.z,
            data.orientation.w,
            data.orientation.x,
            data.orientation.y,
            data.orientation.z,
        ]
        return super().send((data.timestamp, np.array(data_list, dtype=np.float64)), timeout)

    @override
    def receive(self, timeout: Optional[float] = None) -> ImuData:
        """Receive the IMU data."""
        timestamp, data = super().receive(timeout)
        return self.ImuData(
            timestamp=timestamp,
            linear_acceleration=Vec3(x=data[0], y=data[1], z=data[2]),
            angular_velocity=Vec3(x=data[3], y=data[4], z=data[5]),
            orientation=Quat(w=data[6], x=data[7], y=data[8], z=data[9]),
        )


class JointStateCommWrapper(DASHMNumPyComm):
    """Wrapper for the joint state comm."""

    @dataclass
    class JointStateData:
        """Data class for the joint state comm."""

        timestamp: float
        joint_positions: List[float]
        joint_velocities: List[float]
        joint_efforts: List[float]

    def __init__(self, name: str, joint_names: List[str], *, create: bool = True):
        super().__init__(name, (3, len(joint_names)), np.float64, create=create)
        self._joint_names = joint_names  # TODO(haoru): validate the joint names across two sides

    @override
    def send(self, data: JointStateData, timeout: Optional[float] = None) -> bool:
        """Send the joint state data."""
        return super().send(
            (
                data.timestamp,
                np.stack(
                    (data.joint_positions, data.joint_velocities, data.joint_efforts),
                    axis=0,
                ),
            ),
            timeout,
        )

    @property
    def joint_names(self) -> List[str]:
        """Get the joint names."""
        return self._joint_names

    @override
    def receive(self, timeout: Optional[float] = None) -> JointStateData:
        """Receive the joint state data."""
        timestamp, data = super().receive(timeout)
        return self.JointStateData(
            timestamp=timestamp,
            joint_positions=data[0].tolist(),
            joint_velocities=data[1].tolist(),
            joint_efforts=data[2].tolist(),
        )


class JointCommandCommWrapper(DASHMNumPyComm):
    """Wrapper for the joint command comm."""

    @dataclass
    class JointCommandData:
        """Data class for the joint command comm."""

        timestamp: float
        target_positions: List[float]
        target_velocities: List[float]
        target_efforts: List[float]
        stiffness: List[float]
        damping: List[float]

    def __init__(self, name: str, joint_names: List[str], *, create: bool = True):
        super().__init__(name, (5, len(joint_names)), np.float64, create=create)
        self._joint_names = joint_names  # TODO(haoru): validate the joint names across two sides

    @override
    def send(self, data: JointCommandData, timeout: Optional[float] = None) -> bool:
        """Send the joint command data."""
        return super().send(
            (
                data.timestamp,
                np.stack(
                    (
                        data.target_positions,
                        data.target_velocities,
                        data.target_efforts,
                        data.stiffness,
                        data.damping,
                    ),
                    axis=0,
                ),
            ),
            timeout,
        )

    @override
    def receive(self, timeout: Optional[float] = None) -> JointCommandData:
        """Receive the joint command data."""
        timestamp, data = super().receive(timeout)
        return self.JointCommandData(
            timestamp=timestamp,
            target_positions=data[0].tolist(),
            target_velocities=data[1].tolist(),
            target_efforts=data[2].tolist(),
            stiffness=data[3].tolist(),
            damping=data[4].tolist(),
        )

    @property
    def joint_names(self) -> List[str]:
        """Get the joint names."""
        return self._joint_names


class ImageCommWrapper(DASHMNumPyComm):
    """Wrapper for the image comm. It sends raw RGB image in uint8 format."""

    @dataclass
    class ImageData:
        """Data class for the image comm."""

        timestamp: float
        image: np.ndarray

    def __init__(self, name: str, height: int, width: int, *, create: bool = True):
        super().__init__(name, (height, width, 3), np.uint8, create=create)
        self._height = height
        self._width = width

    @override
    def send(self, data: ImageData, timeout: Optional[float] = None) -> bool:
        """Send the image data."""
        return super().send((data.timestamp, data.image), timeout)

    @override
    def receive(self, timeout: Optional[float] = None) -> ImageData:
        """Receive the image data."""
        timestamp, data = super().receive(timeout)
        return self.ImageData(timestamp=timestamp, image=data)

    @property
    def height(self) -> int:
        """Get the height of the image."""
        return self._height

    @property
    def width(self) -> int:
        """Get the width of the image."""
        return self._width


class ProcessedImageCommWrapper(DASHMNumPyComm):
    """Wrapper for the processed image comm."""

    @dataclass
    class ProcessedImageData:
        """Data class for the processed image comm."""

        timestamp: float
        image: np.ndarray

    def __init__(self, name: str, height: int, width: int, dtype: str = "float32", *, create: bool = True):
        super().__init__(name, (height, width, 3), np.dtype(dtype), create=create)

    @override
    def send(self, data: ProcessedImageData, timeout: Optional[float] = None) -> bool:
        """Send the processed image data."""
        return super().send((data.timestamp, data.image), timeout)

    @override
    def receive(self, timeout: Optional[float] = None) -> ProcessedImageData:
        """Receive the processed image data."""
        timestamp, data = super().receive(timeout)
        return self.ProcessedImageData(timestamp=timestamp, image=data)


class ArrayCommWrapper(DASHMNumPyComm):
    """Wrapper for the array comm"""

    @dataclass
    class ArrayData:
        """Data class for the array comm."""

        timestamp: float
        array: np.ndarray

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        *,
        create: bool = True,
    ):
        super().__init__(name, shape, dtype, create=create)

    @override
    def send(self, data: ArrayData, timeout: Optional[float] = None) -> bool:
        """Send the array data."""
        return super().send((data.timestamp, data.array), timeout)

    @override
    def receive(self, timeout: Optional[float] = None) -> ArrayData:
        """Receive the array data."""
        timestamp, data = super().receive(timeout)
        return self.ArrayData(timestamp=timestamp, array=data)


class ObservationEpochCommWrapper(DASHMNumPyComm):
    """Publish-last epoch record for coherent camera and state snapshots."""

    @dataclass(frozen=True)
    class ObservationEpochData:
        """Sequence and capture timing for one fully published observation."""

        observation_seq: int
        publish_time_ns: int
        camera_capture_time_ns: Tuple[int, ...]

    def __init__(self, name: str, camera_count: int, *, create: bool = True):
        if camera_count <= 0:
            raise ValueError("camera_count must be positive")
        super().__init__(name, (camera_count + 2,), np.int64, create=create)
        self._camera_count = camera_count

    @property
    def camera_count(self) -> int:
        """Number of ordered camera timestamps in the epoch."""
        return self._camera_count

    @override
    def send(
        self,
        data: ObservationEpochData,
        timeout: Optional[float] = None,
    ) -> bool:
        """Publish an epoch after all referenced camera and state payloads."""
        camera_times: Sequence[int] = data.camera_capture_time_ns
        if len(camera_times) != self._camera_count:
            raise ValueError("camera capture timestamps must match camera_count")
        if type(data.observation_seq) is not int or data.observation_seq < 0:
            raise ValueError("observation_seq must be non-negative")
        if type(data.publish_time_ns) is not int or data.publish_time_ns <= 0:
            raise ValueError("publish_time_ns must be positive")
        if any(type(timestamp) is not int or timestamp < 0 for timestamp in camera_times):
            raise ValueError("camera capture timestamps must be non-negative integers")
        if max(camera_times) > data.publish_time_ns:
            raise ValueError("camera capture timestamps cannot follow publish_time_ns")
        values = np.asarray(
            [
                data.observation_seq,
                data.publish_time_ns,
                *camera_times,
            ],
            dtype=np.int64,
        )
        return super().send((data.publish_time_ns / 1_000_000_000, values), timeout)

    @override
    def receive(
        self,
        timeout: Optional[float] = None,
    ) -> ObservationEpochData:
        """Receive the latest published observation epoch."""
        _, values = super().receive(timeout)
        return self.ObservationEpochData(
            observation_seq=int(values[0]),
            publish_time_ns=int(values[1]),
            camera_capture_time_ns=tuple(int(value) for value in values[2:]),
        )


class ExecutionProgressCommWrapper(DASHMNumPyComm):
    """Exact chunk identity and completed-row count for phase-locked capture."""

    @dataclass(frozen=True)
    class ExecutionProgressData:
        """One executor start or post-row progress event."""

        timestamp: float
        chunk_capture_time_ns: int
        rows_completed: int
        row_count: int
        action_start_offset_steps: int

    def __init__(self, name: str, *, create: bool = True):
        super().__init__(name, (4,), np.int64, create=create)

    @override
    def send(
        self,
        data: ExecutionProgressData,
        timeout: Optional[float] = None,
    ) -> bool:
        """Publish exact integer progress metadata."""
        if type(data.chunk_capture_time_ns) is not int or data.chunk_capture_time_ns <= 0:
            raise ValueError("chunk_capture_time_ns must be positive")
        if type(data.row_count) is not int or data.row_count <= 0:
            raise ValueError("row_count must be positive")
        if type(data.rows_completed) is not int or not 0 <= data.rows_completed <= data.row_count:
            raise ValueError("rows_completed must be in [0, row_count]")
        if (
            type(data.action_start_offset_steps) is not int
            or data.action_start_offset_steps < 0
            or data.action_start_offset_steps > data.row_count
        ):
            raise ValueError("action_start_offset_steps must be in [0, row_count]")
        values = np.array(
            [
                data.chunk_capture_time_ns,
                data.rows_completed,
                data.row_count,
                data.action_start_offset_steps,
            ],
            dtype=np.int64,
        )
        return super().send((data.timestamp, values), timeout)

    @override
    def receive(
        self,
        timeout: Optional[float] = None,
    ) -> ExecutionProgressData:
        """Receive the latest executor progress event."""
        timestamp, values = super().receive(timeout)
        return self.ExecutionProgressData(
            timestamp=timestamp,
            chunk_capture_time_ns=int(values[0]),
            rows_completed=int(values[1]),
            row_count=int(values[2]),
            action_start_offset_steps=int(values[3]),
        )


class ModeCommWrapper(DASHMNumPyComm):
    """Wrapper for the mode comm."""

    @dataclass
    class ModeData:
        """Data class for the mode comm."""

        timestamp: float
        mode: int

        MODE_0 = 0
        MODE_1 = 1

    def __init__(self, name: str, *, create: bool = True):
        super().__init__(name, (1,), np.int32, create=create)

    @override
    def send(self, data: ModeData, timeout: Optional[float] = None) -> bool:
        """Send the mode data."""
        return super().send((data.timestamp, np.array([data.mode], dtype=np.int32)), timeout)

    @override
    def receive(self, timeout: Optional[float] = None) -> ModeData:
        """Receive the mode data."""
        timestamp, data = super().receive(timeout)
        return self.ModeData(timestamp=timestamp, mode=data[0])


class Nav2DCommWrapper(DASHMNumPyComm):
    """Wrapper for the nav2d comm."""

    @dataclass
    class Nav2DData:
        """Data class for the nav2d comm."""

        timestamp: float
        lin_vel_x: float
        lin_vel_y: float
        ang_vel_z: float

    def __init__(self, name: str, *, create: bool = True):
        super().__init__(name, (3,), np.float64, create=create)

    @override
    def send(self, data: Nav2DData, timeout: Optional[float] = None) -> bool:
        """Send the nav2d data."""
        values = np.array(
            [data.lin_vel_x, data.lin_vel_y, data.ang_vel_z],
            dtype=np.float64,
        )
        return super().send((data.timestamp, values), timeout)

    @override
    def receive(self, timeout: Optional[float] = None) -> Nav2DData:
        """Receive the nav2d data."""
        timestamp, data = super().receive(timeout)
        return self.Nav2DData(timestamp=timestamp, lin_vel_x=data[0], lin_vel_y=data[1], ang_vel_z=data[2])
