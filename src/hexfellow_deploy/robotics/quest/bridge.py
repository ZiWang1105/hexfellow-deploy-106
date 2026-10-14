"""Local HTTPS/WSS bridge between Meta Quest Browser and Policy_Sim."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import hmac
import importlib.resources
import ipaddress
import json
import os
import secrets
import socket
import ssl
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .protocol import ClockSynchronizer, LatestPoseMailbox, QuestPoseSample, decode_pose_packet
from .quality import QuestLinkMonitor

PAIRING_FILE_NAME = "pairing-v1.json"


def _cache_root(cache_dir: Path | None = None) -> Path:
    root = Path.home() / ".cache" / "policy-sim" / "quest-bridge" if cache_dir is None else Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    return root


class PairingStore:
    """Persist one high-entropy browser credential without storing its secret."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.path = _cache_root(cache_dir) / PAIRING_FILE_NAME
        self._lock = threading.Lock()

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @property
    def paired(self) -> bool:
        with self._lock:
            return self._read_digest() is not None

    def authenticate(self, token: str) -> bool:
        if not token:
            return False
        with self._lock:
            expected = self._read_digest()
        return expected is not None and hmac.compare_digest(self._digest(token), expected)

    def pair(self) -> str:
        token = secrets.token_urlsafe(32)
        payload = {"schema_version": 1, "token_sha256": self._digest(token)}
        temporary = self.path.with_name(f"{self.path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
        with self._lock:
            try:
                temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
                temporary.chmod(0o600)
                os.replace(temporary, self.path)
                self.path.chmod(0o600)
            finally:
                temporary.unlink(missing_ok=True)
        return token

    def reset(self) -> None:
        with self._lock:
            self.path.unlink(missing_ok=True)

    def _read_digest(self) -> str | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        digest = payload.get("token_sha256")
        if payload.get("schema_version") != 1 or not isinstance(digest, str) or len(digest) != 64:
            return None
        return digest


def resolve_bind_ip(value: str) -> str:
    """Resolve ``auto`` to the IPv4 address selected by the default route."""

    if value != "auto":
        address = ipaddress.ip_address(value)
        if address.version != 4 or address.is_loopback or address.is_unspecified or not address.is_private:
            raise ValueError("Quest bridge bind IP must be a private, non-loopback IPv4 address")
        return str(address)
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect performs route selection without sending a packet.
        probe.connect(("192.0.2.1", 9))
        selected = probe.getsockname()[0]
    except OSError as error:
        raise RuntimeError("could not detect the Wi-Fi IPv4 address; pass --bind-ip explicitly") from error
    finally:
        probe.close()
    address = ipaddress.ip_address(selected)
    if address.version != 4 or address.is_loopback or address.is_unspecified or not address.is_private:
        raise RuntimeError(f"auto-detected address is not usable by Quest: {selected}")
    return str(address)


def certificate_paths(bind_ip: str, cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Return a persistent self-signed certificate containing an IP SAN."""

    root = _cache_root(cache_dir)
    safe_ip = bind_ip.replace(".", "-")
    certificate = root / f"quest-{safe_ip}.crt"
    private_key = root / f"quest-{safe_ip}.key"
    if certificate.is_file() and private_key.is_file():
        return certificate, private_key
    suffix = f".tmp-{os.getpid()}"
    temporary_certificate = certificate.with_name(certificate.name + suffix)
    temporary_key = private_key.with_name(private_key.name + suffix)
    command = (
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-sha256",
        "-nodes",
        "-days",
        "365",
        "-keyout",
        str(temporary_key),
        "-out",
        str(temporary_certificate),
        "-subj",
        f"/CN={bind_ip}",
        "-addext",
        f"subjectAltName=IP:{bind_ip}",
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        temporary_key.chmod(0o600)
        os.replace(temporary_key, private_key)
        os.replace(temporary_certificate, certificate)
    except FileNotFoundError as error:
        raise RuntimeError("OpenSSL is required to create the Quest HTTPS certificate") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RuntimeError(f"failed to create Quest HTTPS certificate: {detail}") from error
    finally:
        for path in (temporary_certificate, temporary_key):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    return certificate, private_key


class QuestBridge:
    """Own the HTTPS server on an asyncio thread and expose only its latest pose."""

    def __init__(
        self,
        *,
        bind_ip: str = "auto",
        port: int = 8443,
        pin: str | None = None,
        certificate_cache: Path | None = None,
        reset_pairing: bool = False,
        transport: str = "webrtc",
        pose_rate: str = "50",
        hardware_control: bool = False,
        stream_config=None,
    ) -> None:
        if not 1 <= int(port) <= 65535:
            raise ValueError("Quest bridge port must be in [1, 65535]")
        self.bind_ip = resolve_bind_ip(bind_ip)
        self.port = int(port)
        if transport not in {"webrtc", "websocket"}:
            raise ValueError("Quest transport must be webrtc or websocket")
        if pose_rate not in {"30", "50", "60", "90", "auto"}:
            raise ValueError("pose rate must be 30/50/60/90/auto")
        self.pose_rate = pose_rate
        self.hardware_control = hardware_control
        self.stream_config = stream_config
        self.runtime_metrics = {}
        self.transport = transport
        self.server_session_id = secrets.token_hex(16)
        self.pin = f"{secrets.randbelow(1_000_000):06d}" if pin is None else str(pin)
        if len(self.pin) != 6 or not self.pin.isdecimal():
            raise ValueError("Quest bridge PIN must contain exactly six decimal digits")
        self.certificate, self.private_key = certificate_paths(self.bind_ip, certificate_cache)
        self.pairing = PairingStore(certificate_cache)
        if reset_pairing:
            self.pairing.reset()
        self.mailbox = (LatestPoseMailbox() if stream_config is None else LatestPoseMailbox(
            capacity=stream_config.history_capacity, history_ms=stream_config.history_ms))
        self.clock = ClockSynchronizer()
        self.monitor = QuestLinkMonitor()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: Any | None = None
        self._active_websocket: Any | None = None
        self._peer_connection: Any | None = None
        self._negotiation_task: asyncio.Task | None = None
        self._started = threading.Event()
        self._connected = threading.Event()
        self._finish_requested = threading.Event()
        self._metadata_lock = threading.Lock()
        self._client_session_id: str | None = None
        self._client_reference_epoch: int | None = None
        self._client_version: str | None = None
        self._client_stage: str | None = None
        self._client_stage_detail: str | None = None
        self._connection_generation = 0
        self._connected_at_ns: int | None = None
        self._disconnected_at_ns: int | None = None
        self._startup_error: BaseException | None = None
        self._last_client_error: str | None = None
        self._last_pose_feedback_ns = 0
        self._last_pose_sequence: int | None = None
        self._last_pose_epoch: int | None = None
        self._pose_transport_ready = threading.Event()

    @property
    def url(self) -> str:
        return f"https://{self.bind_ip}:{self.port}/"

    @property
    def connected(self) -> bool:
        return self._connected.is_set() and (
            self.transport == "websocket" or self._pose_transport_ready.is_set()
        )

    @property
    def signaling_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def last_client_error(self) -> str | None:
        return self._last_client_error

    @property
    def finish_requested(self) -> bool:
        return self._finish_requested.is_set()

    def clear_finish_request(self) -> None:
        self._finish_requested.clear()

    @property
    def client_session_id(self) -> str | None:
        with self._metadata_lock:
            return self._client_session_id

    @property
    def client_reference_epoch(self) -> int | None:
        with self._metadata_lock:
            return self._client_reference_epoch

    @property
    def client_version(self) -> str | None:
        with self._metadata_lock:
            return self._client_version

    @property
    def client_stage(self) -> str | None:
        with self._metadata_lock:
            return self._client_stage

    @property
    def client_stage_detail(self) -> str | None:
        with self._metadata_lock:
            return self._client_stage_detail

    @property
    def connection_generation(self) -> int:
        with self._metadata_lock:
            return self._connection_generation

    @property
    def connected_at_ns(self) -> int | None:
        with self._metadata_lock:
            return self._connected_at_ns

    @property
    def disconnected_at_ns(self) -> int | None:
        with self._metadata_lock:
            return self._disconnected_at_ns

    def start(self, *, timeout_s: float = 10.0) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main, name="quest-webxr-bridge", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout_s):
            raise TimeoutError("Quest HTTPS bridge did not start in time")
        if self._startup_error is not None:
            raise RuntimeError(f"Quest HTTPS bridge failed to start: {self._startup_error}") from self._startup_error

    def stop(self) -> None:
        loop = self._loop
        thread = self._thread
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None
        self._loop = None
        self._connected.clear()
        self._pose_transport_ready.clear()

    def wait_connected(self, timeout_s: float | None = None) -> bool:
        return self._connected.wait(timeout_s)

    def latest(self) -> QuestPoseSample | None:
        return self.mailbox.get()

    def send_status(self, state: str, detail: str) -> None:
        loop = self._loop
        websocket = self._active_websocket
        if loop is None or websocket is None or websocket.closed or not loop.is_running():
            return
        future = asyncio.run_coroutine_threadsafe(
            websocket.send_json({"type": "status", "state": str(state), "detail": str(detail)}),
            loop,
        )
        future.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)

    def _thread_main(self) -> None:
        try:
            import aiohttp  # noqa: F401
            from aiohttp import web  # noqa: F401
            if self.transport == "webrtc":
                import aiortc  # noqa: F401
        except ImportError:
            self._startup_error = RuntimeError("install the Quest bridge with: python -m pip install -e '.[quest]'")
            self._started.set()
            return
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_start())
            self._started.set()
            loop.run_forever()
        except BaseException as error:
            self._startup_error = error
            self._started.set()
        finally:
            try:
                loop.run_until_complete(self._async_stop())
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    async def _async_start(self) -> None:
        from aiohttp import web

        application = web.Application(client_max_size=4096)
        application.router.add_get("/", self._handle_index)
        application.router.add_get("/ws", self._handle_websocket)
        self._runner = web.AppRunner(application, access_log=None)
        await self._runner.setup()
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.certificate, self.private_key)
        site = web.TCPSite(self._runner, self.bind_ip, self.port, ssl_context=context)
        await site.start()

    async def _async_stop(self) -> None:
        await self._cancel_negotiation()
        if self._peer_connection is not None:
            await self._peer_connection.close()
            self._peer_connection = None
        websocket = self._active_websocket
        if websocket is not None and not websocket.closed:
            await websocket.close(code=1001, message=b"host shutdown")
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _handle_index(self, _request: Any) -> Any:
        from aiohttp import web

        resource = importlib.resources.files("hexfellow_deploy.robotics.quest").joinpath("static/index.html")
        return web.Response(
            text=resource.read_text(encoding="utf-8"),
            content_type="text/html",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    async def _handle_websocket(self, request: Any) -> Any:
        from aiohttp import WSMsgType, web

        expected_origin = f"https://{self.bind_ip}:{self.port}"
        if request.headers.get("Origin") != expected_origin:
            raise web.HTTPForbidden(text="origin rejected")
        if self._active_websocket is not None and not self._active_websocket.closed:
            raise web.HTTPConflict(text="a Quest client is already active")
        websocket = web.WebSocketResponse(heartbeat=5.0, max_msg_size=4096, compress=False)
        await websocket.prepare(request)
        try:
            first = await websocket.receive(timeout=15.0)
            if first.type != WSMsgType.TEXT:
                await websocket.close(code=1008, message=b"PIN required")
                return websocket
            try:
                authentication = json.loads(first.data)
            except json.JSONDecodeError:
                authentication = {}
            if authentication.get("type") != "auth":
                authentication = {}
            supplied_token = str(authentication.get("pairingToken", ""))
            supplied_pin = str(authentication.get("pin", ""))
            session_id = authentication.get("clientSessionId")
            client_version = str(authentication.get("clientVersion", "unknown"))[:128]
            reference_epoch = authentication.get("referenceEpoch", 0)
            if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
                await websocket.send_json(
                    {"type": "auth_error", "code": "invalid_client", "detail": "invalid browser session"}
                )
                await websocket.close(code=1008, message=b"authentication failed")
                return websocket
            try:
                reference_epoch = int(reference_epoch)
            except (TypeError, ValueError):
                reference_epoch = -1
            if reference_epoch < 0:
                await websocket.send_json(
                    {"type": "auth_error", "code": "invalid_client", "detail": "invalid reference epoch"}
                )
                await websocket.close(code=1008, message=b"authentication failed")
                return websocket
            issued_token: str | None = None
            if not self.pairing.authenticate(supplied_token):
                if not hmac.compare_digest(supplied_pin, self.pin):
                    await websocket.send_json(
                        {
                            "type": "auth_error",
                            "code": "pairing_required",
                            "detail": "Pairing expired or PIN incorrect; enter the current six-digit PIN.",
                        }
                    )
                    await websocket.close(code=1008, message=b"authentication failed")
                    return websocket
                issued_token = self.pairing.pair()
            with self._metadata_lock:
                self._client_session_id = session_id
                self._client_reference_epoch = reference_epoch
                self._client_version = client_version
                self._client_stage = "authenticated"
                self._client_stage_detail = "waiting for XR pose frames"
                self._connection_generation += 1
                self._connected_at_ns = time.monotonic_ns()
            self._last_client_error = None
            self._active_websocket = websocket
            self._connected.set()
            self.mailbox.clear()
            self.clock.reset()
            self._last_pose_feedback_ns = 0
            self._last_pose_sequence = None
            self._last_pose_epoch = None
            self._pose_transport_ready.clear()
            response = {
                "type": "auth_ok",
                "preferredFrameRate": 90,
                "serverSessionId": self.server_session_id,
                "poseTransport": self.transport,
                "poseRate": self.pose_rate,
                "hardwareControl": self.hardware_control,
            }
            if issued_token is not None:
                response["pairingToken"] = issued_token
            await websocket.send_json(response)
            sync_task = asyncio.create_task(self._clock_sync_loop(websocket))
            try:
                async for message in websocket:
                    if message.type == WSMsgType.BINARY:
                        await self._accept_pose(message.data, websocket)
                    elif message.type == WSMsgType.TEXT:
                        await self._accept_text(message.data, websocket)
                    elif message.type in {WSMsgType.ERROR, WSMsgType.CLOSE, WSMsgType.CLOSED}:
                        break
            finally:
                sync_task.cancel()
                await asyncio.gather(sync_task, return_exceptions=True)
                await self._cancel_negotiation()
                if self._peer_connection is not None:
                    await self._peer_connection.close()
                    self._peer_connection = None
                self._pose_transport_ready.clear()
        finally:
            if self._active_websocket is websocket:
                self._active_websocket = None
                self._connected.clear()
                self.mailbox.clear()
                with self._metadata_lock:
                    self._disconnected_at_ns = time.monotonic_ns()
        return websocket

    async def _accept_pose(self, payload: bytes, websocket: Any) -> None:
        received_ns = time.monotonic_ns()
        try:
            sample = decode_pose_packet(payload, host_received_ns=received_ns)
        except ValueError as error:
            self._last_client_error = str(error)
            await websocket.send_json({"type": "sample_error", "detail": str(error)})
            return
        estimate = self.clock.estimate()
        if estimate is not None:
            sample = dataclasses.replace(
                sample,
                clock_offset_ms=estimate.host_minus_client_ms,
                clock_uncertainty_ms=estimate.uncertainty_ms,
            )
        if not self._publish_pose(sample):
            return
        if received_ns - self._last_pose_feedback_ns >= 500_000_000:
            self._last_pose_feedback_ns = received_ns
            await websocket.send_json(
                {
                    "type": "pose_ack",
                    "sequence": sample.sequence,
                    "leftPresent": sample.controllers["left"].present,
                    "rightPresent": sample.controllers["right"].present,
                }
            )

    async def _accept_webrtc_offer(self, message: dict[str, Any], websocket: Any) -> None:
        """Negotiate an unordered, no-retransmit LAN pose DataChannel."""

        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

        if self._peer_connection is not None:
            previous_peer = self._peer_connection
            self._peer_connection = None
            await previous_peer.close()
        peer = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        self._peer_connection = peer
        generation = self.connection_generation

        def current():
            return (self._peer_connection is peer and self._active_websocket is websocket
                    and self.connection_generation == generation)

        @peer.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            if not current():
                channel.close()
                return
            if channel.label != "poses" or channel.ordered or channel.maxRetransmits != 0:
                channel.close()
                self._last_client_error = "invalid pose DataChannel reliability options"
                return

            @channel.on("open")
            def on_open() -> None:
                if current():
                    self._pose_transport_ready.set()

            @channel.on("close")
            def on_close() -> None:
                if current():
                    self._pose_transport_ready.clear()
                    self.mailbox.clear()

            # aiortc may deliver the channel after it is already open.
            if channel.readyState == "open":
                self._pose_transport_ready.set()

            @channel.on("message")
            def on_message(payload: Any) -> None:
                if not current() or not isinstance(payload, bytes):
                    return
                received_ns = time.monotonic_ns()
                try:
                    sample = decode_pose_packet(payload, host_received_ns=received_ns)
                except ValueError as error:
                    self._last_client_error = str(error)
                    return
                estimate = self.clock.estimate()
                sample = dataclasses.replace(sample, transport="webrtc")
                if estimate is not None:
                    sample = dataclasses.replace(
                        sample,
                        clock_offset_ms=estimate.host_minus_client_ms,
                        clock_uncertainty_ms=estimate.uncertainty_ms,
                    )
                self._publish_pose(sample)

        @peer.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if current() and peer.connectionState == "failed":
                self._pose_transport_ready.clear()
                self.mailbox.clear()
                try:
                    if not websocket.closed:
                        await websocket.send_json({"type": "webrtc_restart"})
                except (ConnectionError, RuntimeError):
                    # The WSS signaling channel may enter its closing state
                    # between the closed check and the asynchronous write.
                    pass

        offer_sdp = message.get("sdp")
        if not isinstance(offer_sdp, str) or not offer_sdp:
            raise ValueError("invalid WebRTC offer")
        await peer.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        await peer.setLocalDescription(await peer.createAnswer())
        # Gathering completes locally because this LAN mode deliberately uses
        # no external STUN/TURN server.
        while peer.iceGatheringState != "complete":
            await asyncio.sleep(0.01)
        if not current():
            if self._peer_connection is peer:
                self._peer_connection = None
            await peer.close()
            return
        try:
            await websocket.send_json(
                {"type": "webrtc_answer", "sdp": peer.localDescription.sdp}
            )
        except (ConnectionError, RuntimeError):
            if self._peer_connection is peer:
                self._peer_connection = None
            await peer.close()

    async def _cancel_negotiation(self) -> None:
        task = self._negotiation_task
        self._negotiation_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _negotiate(self, message, websocket) -> None:
        completed = False
        try:
            await asyncio.wait_for(self._accept_webrtc_offer(message, websocket), timeout=10.)
            completed = True
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._last_client_error = f"WebRTC negotiation failed: {type(error).__name__}: {error}"
            if not websocket.closed:
                try:
                    await websocket.send_json({"type": "sample_error", "detail": self._last_client_error})
                except (ConnectionError, RuntimeError):
                    # Disconnection can race the closed check above.
                    pass
        finally:
            # A cancelled/failed negotiation must not leave a partial peer or
            # its callbacks able to publish into a subsequent session.
            if not completed:
                peer = self._peer_connection
                self._peer_connection = None
                self._pose_transport_ready.clear()
                self.mailbox.clear()
                if peer is not None:
                    await peer.close()

    def _publish_pose(self, sample: QuestPoseSample) -> bool:
        """Publish only monotonic newest poses; unordered late packets never replace state."""

        epoch = int(sample.reference_epoch)
        sequence = int(sample.sequence) & 0xFFFFFFFF
        if self._last_pose_epoch == epoch and self._last_pose_sequence is not None:
            difference = (sequence - self._last_pose_sequence) & 0xFFFFFFFF
            if difference == 0 or difference >= 0x80000000:
                self.monitor.record_rejection('duplicate' if difference == 0 else 'out_of_order')
                return False
        self._last_pose_epoch = epoch
        self._last_pose_sequence = sequence
        self.mailbox.put(sample)
        self.monitor.record(sample)
        return True

    async def _accept_text(self, payload: str, websocket: Any) -> None:
        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            return
        message_type = message.get("type")
        if message_type == "webrtc_offer" and self.transport == "webrtc":
            await self._cancel_negotiation()
            self._pose_transport_ready.clear()
            self.mailbox.clear()
            self._negotiation_task = asyncio.create_task(self._negotiate(message, websocket))
            return
        if message_type == "client_error":
            self._last_client_error = str(message.get("detail", "unknown client error"))
            return
        if message_type == "client_stage":
            stage = str(message.get("stage", "unknown"))[:128]
            detail = str(message.get("detail", ""))[:512]
            with self._metadata_lock:
                self._client_stage = stage
                self._client_stage_detail = detail
            return
        if message_type == "finish":
            self._finish_requested.set()
            return
        if message_type != "sync_pong":
            return
        try:
            self.clock.add_exchange(
                host_send_ns=int(message["hostSendNs"]),
                client_receive_ms=float(message["clientReceiveMs"]),
                client_send_ms=float(message["clientSendMs"]),
                host_receive_ns=time.monotonic_ns(),
            )
        except (KeyError, TypeError, ValueError):
            self._last_client_error = "invalid clock synchronization response"

    async def _clock_sync_loop(self, websocket: Any) -> None:
        exchanges = 0
        reported_ns = 0
        while not websocket.closed:
            host_send_ns = time.monotonic_ns()
            await websocket.send_json({"type": "sync_ping", "id": exchanges, "hostSendNs": str(host_send_ns)})
            exchanges += 1
            if self.stream_config is not None and host_send_ns - reported_ns >= 1_000_000_000:
                self.runtime_metrics = self.monitor.report(require_static=False,
                    minimum_receive_hz=self.stream_config.minimum_receive_hz).metrics
                reported_ns = host_send_ns
            await asyncio.sleep(0.1 if exchanges < 20 else 1.0)
