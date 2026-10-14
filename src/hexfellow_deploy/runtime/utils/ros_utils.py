from typing import List, Callable, Dict, Union
import logging
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.timer import Timer
from sensor_msgs.msg import Imu, JointState, CompressedImage, Image
from std_msgs.msg import (
    Float32MultiArray,
    Float64MultiArray,
    Int32MultiArray,
    Int64MultiArray,
    MultiArrayLayout,
    MultiArrayDimension,
)

from hexfellow_deploy.runtime.components.comms.wrappers import (
    ImuCommWrapper,
    JointStateCommWrapper,
    JointCommandCommWrapper,
    ImageCommWrapper,
    ProcessedImageCommWrapper,
    ArrayCommWrapper,
)


class ROSMonitorNode(Node):
    """ROS Monitor Node class."""

    def __init__(self, name: str):
        super().__init__(name)

        self._monitor_pubs: Dict[str, Publisher] = {}
        self._monitor_timers: Dict[str, Timer] = {}
        self._monitor_callbacks: Dict[str, Callable] = {}

        self._receive_subs: Dict[str, Subscription] = {}
        self._receive_callbacks: Dict[str, Callable] = {}

    def monitor(
        self,
        wrapper: Union[
            ImuCommWrapper,
            ImageCommWrapper,
            JointStateCommWrapper,
            JointCommandCommWrapper,
            ArrayCommWrapper,
        ],
        interval: float,
    ):
        """Monitor a comm at a given interval."""
        if type(wrapper) == ImuCommWrapper:
            self._imu_handler(wrapper, interval)
        elif type(wrapper) == ImageCommWrapper:
            self._image_handler(wrapper, interval)
        elif type(wrapper) == JointStateCommWrapper:
            self._joint_state_handler(wrapper, interval)
        elif type(wrapper) == JointCommandCommWrapper:
            self._joint_command_handler(wrapper, interval)
        elif type(wrapper) == ArrayCommWrapper:
            self._array_handler(wrapper, interval)
        else:
            raise ValueError(f"Unsupported comm type: {type(wrapper)}")
        logging.getLogger("ROSMonitorNode").info(
            f"Monitoring {wrapper.name} at {1 / interval} Hz"
        )

    def _imu_handler(self, wrapper: ImuCommWrapper, interval: float):
        """IMU handler."""

        def callback(self, wrapper: ImuCommWrapper):
            data = wrapper.receive()
            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.orientation.x = data.orientation.x
            msg.orientation.y = data.orientation.y
            msg.orientation.z = data.orientation.z
            msg.orientation.w = data.orientation.w
            msg.linear_acceleration.x = data.linear_acceleration.x
            msg.linear_acceleration.y = data.linear_acceleration.y
            msg.linear_acceleration.z = data.linear_acceleration.z
            msg.angular_velocity.x = data.angular_velocity.x
            msg.angular_velocity.y = data.angular_velocity.y
            msg.angular_velocity.z = data.angular_velocity.z
            self._monitor_pubs[wrapper.name].publish(msg)

        self._monitor_pubs[wrapper.name] = self.create_publisher(Imu, wrapper.raw_name, 1)
        self._monitor_callbacks[wrapper.name] = lambda: callback(self, wrapper)
        self._monitor_timers[wrapper.name] = self.create_timer(
            interval, self._monitor_callbacks[wrapper.name]
        )

    def _image_handler(self, wrapper: ImageCommWrapper, interval: float):
        """Image handler."""

        def callback(self, wrapper: ImageCommWrapper):
            data = wrapper.receive().image
            assert data.dtype == np.uint8

            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)

            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ok, buf = cv2.imencode(".jpg", data, encode_param)
            if not ok:
                return

            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.format = "jpeg"
            msg.data = buf.tobytes()
            self._monitor_pubs[wrapper.name].publish(msg)

        self._monitor_pubs[wrapper.name] = self.create_publisher(
            CompressedImage, wrapper.raw_name, 1
        )
        self._monitor_callbacks[wrapper.name] = lambda: callback(self, wrapper)
        self._monitor_timers[wrapper.name] = self.create_timer(
            interval, self._monitor_callbacks[wrapper.name]
        )

    def _joint_state_handler(self, wrapper: JointStateCommWrapper, interval: float):
        """Joint state handler."""

        def callback(self, wrapper: JointStateCommWrapper):
            data = wrapper.receive()
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.name = wrapper.joint_names
            msg.position = data.joint_positions
            msg.velocity = data.joint_velocities
            msg.effort = data.joint_efforts
            self._monitor_pubs[wrapper.name].publish(msg)

        self._monitor_pubs[wrapper.name] = self.create_publisher(
            JointState, wrapper.raw_name, 1
        )
        self._monitor_callbacks[wrapper.name] = lambda: callback(self, wrapper)
        self._monitor_timers[wrapper.name] = self.create_timer(
            interval, self._monitor_callbacks[wrapper.name]
        )

    def _joint_command_handler(self, wrapper: JointCommandCommWrapper, interval: float):
        """Joint command handler."""

        def callback(self, wrapper: JointCommandCommWrapper):
            data = wrapper.receive()
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.name = wrapper.joint_names
            msg.position = data.target_positions
            msg.velocity = data.target_velocities
            msg.effort = data.target_efforts
            self._monitor_pubs[wrapper.name].publish(msg)

        self._monitor_pubs[wrapper.name] = self.create_publisher(
            JointState, wrapper.raw_name, 1
        )
        self._monitor_callbacks[wrapper.name] = lambda: callback(self, wrapper)
        self._monitor_timers[wrapper.name] = self.create_timer(
            interval, self._monitor_callbacks[wrapper.name]
        )

    def _array_handler(self, wrapper: ArrayCommWrapper, interval: float):
        """Array handler."""
        msg_cls = Float64MultiArray
        if wrapper.dtype == np.float32:
            msg_cls = Float32MultiArray
        elif wrapper.dtype == np.float64:
            msg_cls = Float64MultiArray
        elif wrapper.dtype == np.int32:
            msg_cls = Int32MultiArray
        elif wrapper.dtype == np.int64:
            msg_cls = Int64MultiArray

        def callback(self, wrapper: ArrayCommWrapper, msg_cls):
            data = wrapper.receive().array
            msg = msg_cls()
            msg.data = data.flatten().tolist()
            for i in range(data.ndim):
                msg.layout.dim.append(
                    MultiArrayDimension(
                        label=f"dim_{i}", size=data.shape[i], stride=data.strides[i]
                    )
                )
            self._monitor_pubs[wrapper.name].publish(msg)

        self._monitor_pubs[wrapper.name] = self.create_publisher(
            msg_cls, wrapper.raw_name, 1
        )
        self._monitor_callbacks[wrapper.name] = lambda: callback(self, wrapper, msg_cls)
        self._monitor_timers[wrapper.name] = self.create_timer(
            interval, self._monitor_callbacks[wrapper.name]
        )

    def receive(self, wrapper: ImageCommWrapper):
        """Receive a message from ROS and communicate with the comm wrapper."""
        if type(wrapper) == ImageCommWrapper:
            self._recv_image_handler(wrapper)
        else:
            raise ValueError(f"Unsupported comm type: {type(wrapper)}")
        logging.getLogger("ROSMonitorNode").info(f"Receiving {wrapper.raw_name} from ROS")

    def _recv_image_handler(self, wrapper: ImageCommWrapper):
        """Image receive handler."""

        def callback(wrapper: ImageCommWrapper, msg: Image | CompressedImage):
            if isinstance(msg, Image):
                data = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3
                )
            elif isinstance(msg, CompressedImage):
                data = np.frombuffer(msg.data, dtype=np.uint8)
                data = cv2.imdecode(data, cv2.IMREAD_COLOR)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

            wrapper.send(
                ImageCommWrapper.ImageData(timestamp=time.monotonic(), image=data)
            )

        self._receive_callbacks[wrapper.name] = lambda msg: callback(wrapper, msg)
        self._receive_subs[wrapper.name] = self.create_subscription(
            Image, wrapper.name, self._receive_callbacks[wrapper.name], 10
        )
