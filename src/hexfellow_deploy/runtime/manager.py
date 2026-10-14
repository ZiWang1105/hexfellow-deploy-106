import logging
import multiprocessing as mp
import os
import time
from multiprocessing.synchronize import Lock as LockType
from typing import Callable, Dict, List

import hydra
from omegaconf import DictConfig
from rich.logging import RichHandler

# rclpy is imported lazily (only when ROS monitor/receive topics are configured)
# so a pure shared-memory deployment does not require a ROS install.
from hexfellow_deploy.runtime.components.base_component import DABaseComponent
from hexfellow_deploy.runtime.components.comms.base_comm import DABaseComm
from hexfellow_deploy.runtime.components.inference.base_inference import DABaseInference
from hexfellow_deploy.runtime.components.input.base_input import DABaseInput
from hexfellow_deploy.runtime.components.interface.base_interface import DABaseInterface
from hexfellow_deploy.runtime.components.output.base_output import DABaseOutput
from hexfellow_deploy.runtime.components.preprocess.base_preprocess import DABasePreprocess
from hexfellow_deploy.runtime.utils.comm_utils import check_locked, set_affinity, set_rt
from hexfellow_deploy.runtime.utils.config_utils import clean_name, register_rl_resolvers

register_rl_resolvers()

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)],
    force=True,
)

# install(show_locals=True, width=200)


class DAManager:
    """Deploy Anything Manager"""

    TIME_BLOCK = 0.000005  # 1 microsecond

    def __init__(
        self,
        topics: Dict[str, DABaseComm],
        interface: dict[str, DABaseInterface],
        preprocess: dict[str, DABasePreprocess],
        input: dict[str, DABaseInput],
        inference: dict[str, DABaseInference],
        output: dict[str, DABaseOutput],
        loop_time: float,
        monitor_topics: Dict[str, float] = {},
        receive_topics: List[str] = [],
        affinities: Dict[str, int | List[int]] = {},
        shutdown_timeout_s: float = 3.0,
    ):
        if shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        self._shutdown_timeout_s = shutdown_timeout_s
        self._interface = interface
        self._preprocess = preprocess
        self._input = input
        self._inference = inference
        self._output = output
        self._loop_time = loop_time
        self._stopping = False
        self._logger = logging.getLogger("DAManager")
        self._monitor_topics = monitor_topics
        self._receive_topics = receive_topics
        self._affinities = affinities

        # Every topic is a single shared comm instance (the registry owns it, and
        # components reference the same object via ${topic:...}). No priority
        # ordering: each topic creates its own segment independently.
        self._comms: List[DABaseComm] = list(topics.values())
        self._comm_by_name: Dict[str, DABaseComm] = {c.name: c for c in self._comms}
        self._all_components: List[DABaseComponent] = (
            list(self._interface.values())
            + list(self._preprocess.values())
            + list(self._input.values())
            + list(self._inference.values())
            + list(self._output.values())
        )

        # compute the time budget for each stage of the pipeline
        if len(self._preprocess) > 0:
            self._preprocess_time_budgets = [component.time_budget for component in self._preprocess.values()]
            self._max_preprocess_time_budget = max(self._preprocess_time_budgets)
            self._preprocess_idx_by_time_budget = sorted(
                range(len(self._preprocess_time_budgets)),
                key=lambda x: self._preprocess_time_budgets[x],
                reverse=True,
            )
            self._preprocess_name_by_time_budget = [
                list(self._preprocess.keys())[idx] for idx in self._preprocess_idx_by_time_budget
            ]
            self._logger.info(f"Max preprocess time budget: {self._max_preprocess_time_budget} s")
        else:
            self._preprocess_name_by_time_budget = []
            self._max_preprocess_time_budget = 0.0
            self._preprocess_idx_by_time_budget = []
            self._preprocess_name_by_time_budget = []
            self._logger.info("No preprocess components found.")

        if len(self._input) > 0:
            self._input_time_budgets = [component.time_budget for component in self._input.values()]
            self._max_input_time_budget = max(self._input_time_budgets)
            self._input_idx_by_time_budget = sorted(
                range(len(self._input_time_budgets)),
                key=lambda x: self._input_time_budgets[x],
                reverse=True,
            )
            self._input_name_by_time_budget = [list(self._input.keys())[idx] for idx in self._input_idx_by_time_budget]
            self._logger.info(f"Max input time budget: {self._max_input_time_budget} s")
        else:
            self._input_name_by_time_budget = []
            self._max_input_time_budget = 0.0
            self._input_idx_by_time_budget = []
            self._input_name_by_time_budget = []
            self._logger.info("No input components found.")

        if len(self._inference) > 0:
            self._inference_time_budgets = [component.time_budget for component in self._inference.values()]
            self._max_inference_time_budget = max(self._inference_time_budgets)
            self._inference_idx_by_time_budget = sorted(
                range(len(self._inference_time_budgets)),
                key=lambda x: self._inference_time_budgets[x],
                reverse=True,
            )
            self._inference_name_by_time_budget = [
                list(self._inference.keys())[idx] for idx in self._inference_idx_by_time_budget
            ]
            self._logger.info(f"Max inference time budget: {self._max_inference_time_budget} s")
        else:
            self._inference_name_by_time_budget = []
            self._max_inference_time_budget = 0.0
            self._inference_idx_by_time_budget = []
            self._inference_name_by_time_budget = []
            self._logger.info("No inference components found.")

        if len(self._output) > 0:
            self._output_time_budgets = [component.time_budget for component in self._output.values()]
            self._max_output_time_budget = max(self._output_time_budgets)
            self._output_idx_by_time_budget = sorted(
                range(len(self._output_time_budgets)),
                key=lambda x: self._output_time_budgets[x],
                reverse=True,
            )
            self._output_name_by_time_budget = [
                list(self._output.keys())[idx] for idx in self._output_idx_by_time_budget
            ]
            self._logger.info(f"Max output time budget: {self._max_output_time_budget} s")

        else:
            self._output_name_by_time_budget = []
            self._max_output_time_budget = 0.0
            self._output_idx_by_time_budget = []
            self._output_name_by_time_budget = []
            self._logger.info("No output components found.")

        self._logger.info(
            "Total pipeline time budget: %s s",
            self._max_preprocess_time_budget
            + self._max_input_time_budget
            + self._max_inference_time_budget
            + self._max_output_time_budget,
        )

        # assign locks of every stage
        self._preprocess_start_locks = [mp.Lock() for _ in range(len(self._preprocess_name_by_time_budget))]
        self._preprocess_end_locks = [mp.Lock() for _ in range(len(self._preprocess_name_by_time_budget))]
        for i, name in enumerate(self._preprocess_name_by_time_budget):
            start_lock = self._preprocess_start_locks[i]
            start_lock.acquire()
            end_lock = self._preprocess_end_locks[i]
            # end_lock.acquire()
            self._preprocess[name].set_start_processing_lock(start_lock)
            self._preprocess[name].set_end_processing_lock(end_lock)

        self._input_start_locks = [mp.Lock() for _ in range(len(self._input_name_by_time_budget))]
        self._input_end_locks = [mp.Lock() for _ in range(len(self._input_name_by_time_budget))]
        for i, name in enumerate(self._input_name_by_time_budget):
            start_lock = self._input_start_locks[i]
            start_lock.acquire()
            end_lock = self._input_end_locks[i]
            # end_lock.acquire()
            self._input[name].set_start_gathering_lock(start_lock)
            self._input[name].set_end_gathering_lock(end_lock)

        self._inference_start_locks = [mp.Lock() for _ in range(len(self._inference_name_by_time_budget))]
        self._inference_end_locks = [mp.Lock() for _ in range(len(self._inference_name_by_time_budget))]
        for i, name in enumerate(self._inference_name_by_time_budget):
            start_lock = self._inference_start_locks[i]
            start_lock.acquire()
            end_lock = self._inference_end_locks[i]
            # end_lock.acquire()
            self._inference[name].set_start_inference_lock(start_lock)
            self._inference[name].set_end_inference_lock(end_lock)

        self._output_start_locks = [mp.Lock() for _ in range(len(self._output_name_by_time_budget))]
        self._output_end_locks = [mp.Lock() for _ in range(len(self._output_name_by_time_budget))]
        for i, name in enumerate(self._output_name_by_time_budget):
            start_lock = self._output_start_locks[i]
            start_lock.acquire()
            end_lock = self._output_end_locks[i]
            # end_lock.acquire()
            self._output[name].set_start_output_lock(start_lock)
            self._output[name].set_end_output_lock(end_lock)

        # Comms own their own per-topic named semaphore (created in initialize),
        # so there is no lock to assign here — the same named semaphore is shared
        # by fork and openable by name from an external attach() consumer.

        # add ROS monitoring/receiving to comms (only if any topic is requested,
        # so rclpy is not required for a pure shared-memory deployment)
        if monitor_topics or receive_topics:
            from hexfellow_deploy.runtime.utils.ros_utils import ROSMonitorNode

            self._monitor_node = ROSMonitorNode("monitor_node")
            for topic, interval in (monitor_topics or {}).items():
                topic_cleaned = clean_name(topic)
                if topic_cleaned not in self._comm_by_name:
                    raise ValueError(f"Topic {topic} not found in communication channels.")
                self._monitor_node.monitor(self._comm_by_name[topic_cleaned], interval)
            for topic in receive_topics or []:
                topic_cleaned = clean_name(topic)
                if topic_cleaned not in self._comm_by_name:
                    raise ValueError(f"Topic {topic} not found in communication channels.")
                self._monitor_node.receive(self._comm_by_name[topic_cleaned])
        else:
            self._monitor_node = None

        # initialize comms
        for comm in self._comms:
            comm.initialize()

        # initialize components
        for component in self._all_components:
            component.initialize()

        # find unused cores and assign to processes
        all_cores = list(range(os.cpu_count()))
        used_cores = []
        for affinity in self._affinities.values():
            if isinstance(affinity, int):
                used_cores.append(affinity)
            elif isinstance(affinity, list):
                used_cores.extend(affinity)
        unused_cores = [core for core in all_cores if core not in used_cores]
        if len(unused_cores) == 0:
            unused_cores = None

        # run components
        self._processes: List[mp.Process] = []
        self._processes.append(
            mp.Process(
                target=self._run_with_priority,
                args=("manager", self._run, 10, self._affinities.get("manager", unused_cores)),
                daemon=True,
                name="manager",
            )
        )
        # Comms get no process of their own: DASHMComm.run() is a no-op, the
        # segments are created in the parent, and the component processes share
        # them by fork. Spawning per-comm processes would only fork the address
        # space to do nothing (and widen the resource_tracker blast radius).
        for component in self._all_components:
            self._processes.append(
                mp.Process(
                    target=self._run_with_priority,
                    args=(
                        component.name,
                        component.run,
                        10,
                        self._affinities.get(component.name, None),
                    ),
                    daemon=True,
                    name=component.name,
                )
            )

    def run_all(self) -> None:
        for process in self._processes:
            process.start()

        try:
            if self._monitor_node:
                import rclpy

                rclpy.spin(self._monitor_node)
            else:
                while True:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            self._logger.info("Keyboard interrupt received. Stopping components...")
            for component in self._all_components:
                self._logger.info(f"Signaling stop to {component.name}")
                component.signal_stop()
            for comm in self._comms:
                self._logger.info(f"Signaling stop to {comm.name}")
                comm.signal_stop()
            self._stopping = True
            for process in self._processes:
                self._logger.info(f"Joining process {process.name}")
                process.join(timeout=self._shutdown_timeout_s)
                if process.is_alive():
                    self._logger.warning(f"Component {process.name} did not stop in time. Terminating.")
                    process.terminate()
            for component in self._all_components:
                self._logger.info(f"Cleaning up {component.name}")
                component.clean_up()
            for comm in reversed(self._comms):
                self._logger.info(f"Cleaning up {comm.name}")
                comm.clean_up()
            self._logger.info("All components cleaned up.")

    def _run(self) -> None:
        """Run the manager."""
        we_own_preprocess_locks = [True] * len(self._preprocess_name_by_time_budget)
        we_own_input_locks = [True] * len(self._input_name_by_time_budget)
        we_own_inference_locks = [True] * len(self._inference_name_by_time_budget)
        we_own_output_locks = [True] * len(self._output_name_by_time_budget)

        next_target_time = (
            time.monotonic()
            + self._max_preprocess_time_budget
            + self._max_input_time_budget
            + self._max_inference_time_budget
            + self._max_output_time_budget
        )
        while not self._stopping:
            self._step_components()

            for i, name in enumerate(self._preprocess_name_by_time_budget):
                # sleep until the next preprocess start time
                next_preprocess_start_time = (
                    next_target_time
                    - self._max_output_time_budget
                    - self._max_inference_time_budget
                    - self._max_input_time_budget
                    - self._preprocess[name].time_budget
                )
                self._sleep_until(lambda: time.monotonic() >= next_preprocess_start_time)
                # unlock the preprocess start lock
                if we_own_preprocess_locks[i]:
                    self._preprocess[name].start_processing_lock.release()
                    we_own_preprocess_locks[i] = False

            # sleep until the first input start time
            next_input_start_time = (
                next_target_time
                - self._max_output_time_budget
                - self._max_inference_time_budget
                - self._max_input_time_budget
            )
            self._sleep_until(lambda: time.monotonic() >= next_input_start_time)
            # check all the preprocess start locks are released
            for i, name in enumerate(self._preprocess_name_by_time_budget):
                if check_locked(self._preprocess[name].start_processing_lock):
                    self._logger.warning(
                        f"Preprocess {name} is not ready in time budget {self._preprocess[name].time_budget} s."
                    )

            for i, name in enumerate(self._input_name_by_time_budget):
                # sleep until the next input start time
                next_input_start_time = (
                    next_target_time
                    - self._max_output_time_budget
                    - self._max_inference_time_budget
                    - self._input[name].time_budget
                )
                self._sleep_until(lambda: time.monotonic() >= next_input_start_time)
                # unlock the input start lock
                if we_own_input_locks[i]:
                    self._input[name].start_gathering_lock.release()
                    we_own_input_locks[i] = False

            # sleep until the first inference start time
            next_inference_start_time = (
                next_target_time - self._max_output_time_budget - self._max_inference_time_budget
            )
            self._sleep_until(lambda: time.monotonic() >= next_inference_start_time)
            # check all the input start locks are released
            for i, name in enumerate(self._input_name_by_time_budget):
                if check_locked(self._input[name].start_gathering_lock):
                    self._logger.warning(f"Input {name} is not ready in time budget {self._input[name].time_budget} s.")

            for i, name in enumerate(self._inference_name_by_time_budget):
                # sleep until the next inference start time
                next_inference_start_time = (
                    next_target_time - self._max_output_time_budget - self._inference[name].time_budget
                )
                self._sleep_until(lambda: time.monotonic() >= next_inference_start_time)
                # unlock the inference start lock
                if we_own_inference_locks[i]:
                    self._inference[name].start_inference_lock.release()
                    we_own_inference_locks[i] = False

            # sleep until the first output start time
            next_output_start_time = next_target_time - self._max_output_time_budget
            self._sleep_until(lambda: time.monotonic() >= next_output_start_time)
            # check all the inference start locks are released
            for i, name in enumerate(self._inference_name_by_time_budget):
                if check_locked(self._inference[name].start_inference_lock):
                    self._logger.warning(
                        f"Inference {name} is not ready in time budget {self._inference[name].time_budget} s."
                    )

            for i, name in enumerate(self._output_name_by_time_budget):
                # sleep until the next output start time
                next_output_start_time = next_target_time - self._output[name].time_budget
                self._sleep_until(lambda: time.monotonic() >= next_output_start_time)
                # unlock the output start lock
                if we_own_output_locks[i]:
                    try:
                        self._output[name].start_output_lock.release()
                    except Exception:
                        pass
                    we_own_output_locks[i] = False

            # sleep until the target time
            self._sleep_until(lambda: time.monotonic() >= next_target_time)
            # check all the output end locks are released
            for i, name in enumerate(self._output_name_by_time_budget):
                if check_locked(self._output[name].start_output_lock):
                    self._logger.warning(
                        f"Output {name} is not ready in time budget of {self._output[name].time_budget} s."
                    )

            # set the next target time
            next_target_time = time.monotonic() + self._loop_time
            next_preprocess_start_time = (
                next_target_time
                - self._max_output_time_budget
                - self._max_inference_time_budget
                - self._max_input_time_budget
                - self._max_preprocess_time_budget
            )

            # until then, we need to lock everything back
            for i, (name, preprocess) in enumerate(self._preprocess.items()):
                if not we_own_preprocess_locks[i]:
                    if self._try_lock_until(preprocess.start_processing_lock, next_preprocess_start_time):
                        we_own_preprocess_locks[i] = True
                    else:
                        self._logger.warning(f"Preprocess {name} is not ready before the next cycle.")

            for i, (name, input) in enumerate(self._input.items()):
                if not we_own_input_locks[i]:
                    if self._try_lock_until(input.start_gathering_lock, next_input_start_time):
                        we_own_input_locks[i] = True
                    else:
                        self._logger.warning(f"Input {name} is not ready before the next cycle.")

            for i, (name, inference) in enumerate(self._inference.items()):
                if not we_own_inference_locks[i]:
                    if self._try_lock_until(inference.start_inference_lock, next_inference_start_time):
                        we_own_inference_locks[i] = True
                    else:
                        self._logger.warning(f"Inference {name} is not ready before the next cycle.")

            for i, (name, output) in enumerate(self._output.items()):
                if not we_own_output_locks[i]:
                    if self._try_lock_until(output.start_output_lock, next_output_start_time):
                        we_own_output_locks[i] = True
                    else:
                        self._logger.warning(f"Output {name} is not ready before the next cycle.")

    def _run_with_priority(
        self,
        name: str,
        func: Callable[[], None],
        priority: int = 10,
        affinity: int | List[int] = None,
    ) -> None:
        set_rt(policy="rr", prio=priority)
        if affinity is not None:
            set_affinity(affinity)
        try:
            func()
        except Exception as e:
            logging.getLogger(name).error(f"Exception when running component {name}")
            raise e

    def _sleep_until(self, func: Callable[[], bool]) -> float:
        """Sleep until the function returns True."""
        total_sleep_time = 0.0
        while not func():
            time.sleep(self.TIME_BLOCK)
            total_sleep_time += self.TIME_BLOCK
        return total_sleep_time

    def _try_lock_until(self, lock: LockType, time_point: float) -> bool:
        """Try to lock the lock until the time is up."""
        return lock.acquire(timeout=time_point - time.monotonic())

    def _step_components(self):
        """Step the components."""
        for component in self._all_components:
            component.tick()


@hydra.main(
    config_path="../../configs/exp",
    version_base=None,
)
def main(config: DictConfig):
    from hexfellow_deploy.runtime.utils.config_utils import (
        build_topic_registry,
        register_topic_resolver,
    )

    # Instantiate every topic once, then expose them via ${topic:<name>} so the
    # manager's component tree resolves each reference to the same shared comm.
    registry = build_topic_registry(config.topics)
    register_topic_resolver(registry)

    manager_cfg = config.manager
    use_ros = bool(manager_cfg.get("monitor_topics")) or bool(manager_cfg.get("receive_topics"))
    if use_ros:
        import rclpy

        rclpy.init()
    manager: DAManager = hydra.utils.instantiate(manager_cfg, topics=registry)
    manager.run_all()
    if use_ros:
        import rclpy

        rclpy.shutdown()


if __name__ == "__main__":
    main()
