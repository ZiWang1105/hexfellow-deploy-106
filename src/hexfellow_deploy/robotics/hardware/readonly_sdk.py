"""Read-only downlink allowlist installed before SDK connection startup."""

from .x7 import HardwareSafetyError


def readonly_api_type(base_type):
    class ReadOnlyApi(base_type):
        def __init__(self, *args, **kwargs):
            self.blocked_control_messages = 0
            super().__init__(*args, **kwargs)

        async def _send_down_message(self, data):
            kind = data.WhichOneof("down")
            if kind not in {"set_report_frequency", "enable_kcp", "placeholder_message"}:
                self.blocked_control_messages += 1
                raise HardwareSafetyError(f"read-only SDK blocked downstream message: {kind}")
            return await super()._send_down_message(data)

        async def _device_periodic_runner(self, device_id):
            from hex_device.hands import Hands

            device = self._device_id_map.get(device_id)
            if isinstance(device, Hands):
                # SDK 1.4.10 subtracts an unset _last_command_time on passive
                # hands and then retries every 0.5 s. Hands._init() is a no-op;
                # this periodic task only sends commands. Telemetry is updated
                # independently by _process_api_up, which remains untouched.
                return
            return await super()._device_periodic_runner(device_id)

    return ReadOnlyApi
