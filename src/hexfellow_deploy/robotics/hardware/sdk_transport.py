"""Instance-local scheduling/UDP boundary for the pinned Hex SDK (1.4.10).

The existing SDK periodic task remains the only sender. No SDK module globals
or installed files are modified. UDP return is host submission, never peer ACK.
"""
import asyncio
import ipaddress
import socket
import time
import types
from collections import deque

from .x7 import HardwareSafetyError


def immediate_api_type(base):
    class ImmediateApi(base):
        async def _device_periodic_runner(self, device_id):
            device = self._device_id_map.get(device_id)
            if device is not None:
                event = asyncio.Event()
                loop = asyncio.get_running_loop()
                device._policy_wake = lambda: loop.call_soon_threadsafe(event.set)
                original = device._periodic.__func__

                async def delay(start, period_ms):
                    remaining = max(0., period_ms / 1000 - (time.perf_counter() - start))
                    try:
                        if not event.is_set():
                            await asyncio.wait_for(event.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        pass
                    event.clear()

                if 'delay' not in original.__globals__:
                    raise HardwareSafetyError('unsupported SDK periodic scheduler')
                namespace = dict(original.__globals__, delay=delay)
                adapted = types.FunctionType(original.__code__, namespace, original.__name__,
                                             original.__defaults__, original.__closure__)
                device._periodic = types.MethodType(adapted, device)
            return await super()._device_periodic_runner(device_id)
    return ImmediateApi


class CheckedUDP:
    """Resolve once and latch errors even when the C/SDK wrapper swallows them."""
    def __init__(self, client, gate=None):
        self.client, self.gate = client, gate
        self.identity = (client.server_address, client.server_port, client._sock, getattr(client, "_kcp", None))
        address, port, *_ = self.identity
        host, _, zone = address.partition('%')
        scope = (int(zone) if zone.isdecimal() else socket.if_nametoindex(zone)) if zone else 0
        try:
            parsed = ipaddress.ip_address(host)
            host = f'::ffff:{host}' if parsed.version == 4 else host
            self.destination = (host, port, 0, scope)
        except ValueError:
            self.destination = socket.getaddrinfo(host, port, socket.AF_INET6, socket.SOCK_DGRAM)[0][4]
        self.fault = None
        self.connection_started_at = time.monotonic()
        self.context = None  # Accessed under the SDK's reentrant KCP lock.
        self.rows = deque(maxlen=2048)
        self.receive_rows = deque(maxlen=2048)
        self.last_receive = None
        self.segments = {}
        self.unacked = {}
        self.acknowledged_hold_epoch = -1
        self.pending_frames = {}  # Frame attribution survives a deferred KCP flush.
        self.pending_count = 0
        self.stop_ack = None
        original_receive = client._receive_from_socket
        self.kernel_timestamps = False
        try:
            import sys
            if sys.platform == "linux":
                client._sock.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_TIMESTAMPNS", 35), 1)
                self.kernel_timestamps = True
        except (AttributeError, OSError):
            pass

        def receive():
            kernel_wall_ns = None
            if self.kernel_timestamps:
                import struct
                try:
                    data, ancillary, flags, address = client._sock.recvmsg(client._config.max_message_length, 128)
                    if flags & socket.MSG_TRUNC:
                        self.fault = 'truncated UDP telemetry'
                        return None
                    expected = client.filter_address
                    host = address[0]
                    if expected is not None and (address[1] != expected[1] or
                            host != expected[0] and host.removeprefix('::ffff:') != expected[0]):
                        return None
                    for level, kind, value in ancillary:
                        if level == socket.SOL_SOCKET and kind == getattr(socket, "SO_TIMESTAMPNS", 35):
                            seconds, ns = struct.unpack('@ll', value[:struct.calcsize('@ll')])
                            kernel_wall_ns = seconds * 1000000000 + ns
                except socket.timeout:
                    return None
                except OSError as error:
                    if client._running:
                        self.fault = f'UDP receive failed: {error}'
                    return None
            else:
                data = original_receive()
            if data is not None:
                self.last_receive = time.monotonic()
                row = {'udp_received_at': self.last_receive, 'bytes': len(data),
                       'clock': 'host_monotonic_after_recvfrom',
                       'kernel_timestamp_available': kernel_wall_ns is not None}
                import struct
                segments = []
                offset = 0
                while offset + 24 <= len(data):
                    conv, cmd, fragment, window, stamp, sn, una, size = struct.unpack_from('<IBBHIIII', data, offset)
                    if offset + 24 + size > len(data):
                        break
                    segments.append((conv, cmd, sn, una, stamp, fragment))
                    offset += 24 + size
                row['kcp_segments_conv_cmd_sn_una_ts_fragment'] = segments
                if kernel_wall_ns is not None:
                    before = time.monotonic_ns()
                    wall = time.time_ns()
                    after = time.monotonic_ns()
                    row.update(kernel_realtime_ns=kernel_wall_ns,
                               kernel_monotonic_estimate=(kernel_wall_ns-wall+(before+after)//2)/1e9,
                               clock_sampling_uncertainty_ns=(after-before)//2,
                               clock_adjustments_bounded=False)
                self.last_receive_row = row
                self.receive_rows.append(row)
            return data

        original_process = client._process_received_data
        self.process_rows = deque(maxlen=2048)
        def process(data):
            self.process_entered = time.monotonic()
            try:
                result = original_process(data)
                # Cumulative UNA is a receive-order barrier, not an execution ACK.
                import struct
                with client._kcp_lock:
                    offset = 0
                    while offset + 24 <= len(data):
                        conv, cmd, _, _, _, _, una, size = struct.unpack_from('<IBBHIIII', data, offset)
                        if offset + 24 + size > len(data):
                            break
                        if cmd in {81, 82, 83, 84}:
                            for sn, source in tuple(self.unacked.items()):
                                if source and source.get('conv') == conv and 0 < ((una - sn) & 0xffffffff) < 0x80000000:
                                    source['acknowledged'] = True
                                    self.unacked.pop(sn, None)
                                    if source.get('hold'):
                                        self.acknowledged_hold_epoch = max(
                                            self.acknowledged_hold_epoch, source['recovery_epoch'])
                                    if source.get('stop_token') is not None and self.stop_ack is None:
                                        self.stop_ack = dict(stop_token=source['stop_token'], conv=conv, sn=sn,
                                            cumulative_una=una, acknowledged_at=time.monotonic(),
                                            connection_started_at=self.connection_started_at)
                        offset += 24 + size
                return result
            finally:
                self.process_rows.append((self.process_entered, time.monotonic(), len(data)))
        client._process_received_data = process
        client._receive_from_socket = receive
        client._send_raw_data = self.send
        client._policy_udp = self

    def motion_drained(self):
        with self.client._kcp_lock:
            return (not any(not s.get('hold') for rows in self.pending_frames.values() for s in rows)
                    and all(s.get('hold') for s in self.unacked.values()))

    def enqueue_context(self, frame, context):
        if context is not None:
            if self.pending_count >= 512:
                raise HardwareSafetyError('KCP frame attribution capacity exceeded')
            # Pinned control packets must fit one KCP fragment. Fragmented
            # protocol traffic is never treated as a new authorized motion.
            if len(frame) > 1200:
                raise HardwareSafetyError('control frame exceeds single-fragment contract')
            self.pending_frames.setdefault(bytes(frame), deque()).append(dict(context))
            self.pending_count += 1

    def hold_pending(self, epoch):
        return (any(s.get('hold') and s.get('recovery_epoch') == epoch
                    for rows in self.pending_frames.values() for s in rows)
                or any(s and s.get('hold') and s.get('recovery_epoch') == epoch and not s.get('acknowledged')
                       for s in self.unacked.values()))

    def hold_confirmed(self, epoch):
        with self.client._kcp_lock:
            return self.acknowledged_hold_epoch == epoch

    def stop_evidence(self, requested):
        with self.client._kcp_lock:
            unchanged = self.identity == (self.client.server_address, self.client.server_port,
                                          self.client._sock, getattr(self.client, '_kcp', None))
            ack = self.stop_ack
            valid = bool(unchanged and not self.fault and ack and self.gate is not None
                         and not self.gate.fault and self.gate.phase == 'stopping'
                         and ack['stop_token'] == self.gate.stop_token
                         and ack['stop_token'] >= requested and self.motion_drained())
            return dict(ack or {}, brake_received=valid, no_pending_motion=valid)

    def fail(self, error):
        self.fault = self.fault or str(error)
        if self.gate is not None:
            self.gate.fault = self.gate.fault or f'UDP send failed: {self.fault}'
        raise HardwareSafetyError(self.fault)

    def send(self, data):
        # Serialize terminal transition with the actual socket submission, not
        # just the earlier SDK filtering. Lock order is KCP -> gate throughout.
        if self.gate is not None:
            with self.gate.lock:
                return self._send_locked(data)
        return self._send_locked(data)

    def _send_locked(self, data):
        import struct
        try:
            if self.identity != (self.client.server_address, self.client.server_port, self.client._sock, getattr(self.client, "_kcp", None)):
                raise HardwareSafetyError('KCP connection changed during control')
            # KCP ACK/WASK/WINS are transport traffic, not new motion sends.
            context = None
            now = time.monotonic()
            offset = 0
            retransmission = False
            rewritten = []
            expired = None
            segment_evidence = []
            while offset + 24 <= len(data):
                conv, _, _, _, stamp = struct.unpack_from('<IBBHI', data, offset)
                cmd = data[offset + 4]
                sn, _, size = struct.unpack_from('<III', data, offset + 12)
                if cmd == 81:
                    if sn in self.segments:
                        retransmission = True
                        source = self.segments[sn]
                        if source is not None and (source.get('braked') or now >= source['lease_deadline']):
                            expired = 'expired motion KCP retransmission replaced with brake'
                    else:
                        payload = bytes(data[offset+24:offset+24+size])
                        pending = self.pending_frames.get(payload)
                        if pending:
                            source = pending.popleft()
                            self.pending_count -= 1
                            if not pending:
                                del self.pending_frames[payload]
                        else:
                            source = dict(self.context) if self.context is not None else None
                        if source is not None:
                            source['conv'] = conv
                        self.segments[sn] = source
                        if source is not None:
                            self.unacked[sn] = source
                        # KCP window is 128; retain far more than outstanding data.
                        if len(self.segments) > 4096:
                            retired = next((key for key, value in self.segments.items()
                                            if value is None or value.get('acknowledged')), None)
                            if retired is None:
                                raise HardwareSafetyError('KCP outstanding segment accounting overflow')
                            del self.segments[retired]
                        if source is not None:
                            context = self.context if self.context is not None and source.get('sequence') == self.context.get('sequence') else source
                            if now >= source['deadline']:
                                expired = 'command expired before actual UDP send'
                    recovery_hold = source is not None and not source.get('hold') and self.gate is not None and self.gate.hold_resume and (
                        self.gate.phase == 'holding' or source.get('recovery_epoch', 0) != self.gate.recovery_epoch)
                    terminal_stop = (source is not None and not source.get('hold') and self.gate is not None
                                     and self.gate.phase == 'stopping')
                    if recovery_hold:
                        expired = 'old recovery epoch motion replaced with brake'
                    if terminal_stop:
                        expired = 'pending motion replaced with brake after stop'
                    if source is not None and (terminal_stop or recovery_hold or source.get('braked') or now >= source['lease_deadline'] or
                            context is not None and now >= source['deadline']):
                        # Do not leave a missing KCP sequence: that would block all
                        # subsequent brakes behind reliable-stream ordering. Replace
                        # the unsent single-fragment motion with a brake at that SN.
                        if self.gate is None or data[offset + 5] != 0:
                            raise HardwareSafetyError('cannot replace fragmented expired motion')
                        from hex_device.generated.public_api_down_pb2 import APIDown
                        from hex_device.hex_device_api import HexSocketOpcode, HexSocketParser

                        from .mit_wire import brake_packet
                        source['braked'] = True
                        brake = brake_packet(self.gate.name, APIDown())
                        brake.protocol_major_version, brake.protocol_minor_version = 1, 6
                        frame = HexSocketParser.create_header(brake.SerializeToString(), HexSocketOpcode.Binary)
                        rewritten.append(data[offset:offset+20] + struct.pack('<I', len(frame)) + frame)
                    else:
                        rewritten.append(data[offset:offset+24+size])
                else:
                    rewritten.append(data[offset:offset+24+size])
                offset += 24 + size
                source = self.segments.get(sn) if cmd == 81 else None
                segment_evidence.append({
                    'conv': conv, 'sn': sn, 'cmd': cmd, 'stamp': stamp,
                    'source_sequence': source.get('sequence') if source else None,
                    'deadline': source.get('deadline') if source and not source.get('hold') else None,
                    'lease_deadline': source.get('lease_deadline') if source and not source.get('hold') else None,
                    'replaced_with_brake': bool(source and source.get('braked')),
                })
            if expired is not None:
                # Once any motion expires, a newer motion in the same UDP flush
                # must not follow the replacement brake and re-enable movement.
                for index, segment in enumerate(rewritten):
                    if segment[4] != 81:
                        continue
                    sn = struct.unpack_from('<I', segment, 12)[0]
                    source = self.segments.get(sn)
                    if source is not None:
                        if source.get('hold'):
                            continue
                        if segment[5] != 0:
                            raise HardwareSafetyError('cannot replace fragmented motion during stop')
                        source['braked'] = True
                        rewritten[index] = segment[:20] + struct.pack('<I', len(frame)) + frame
                        segment_evidence[index]['replaced_with_brake'] = True
                data = b''.join(rewritten)
            if len(data) > self.client._config.max_message_length:
                raise HardwareSafetyError('oversized KCP datagram')
            sent = self.client._sock.sendto(data, self.destination)
            if sent != len(data):
                raise HardwareSafetyError('short UDP send')
            row = dict(udp_send_started=now, udp_send_returned=time.monotonic(),
                       bytes=sent, sequence=context['sequence'] if context else None,
                       kind='replacement_brake' if expired else ('command_flush' if context else ('retransmission' if retransmission else 'transport')),
                       hardware_acknowledged=False)
            row['segments'] = segment_evidence
            row['expiry_reason'] = expired
            row['connection_started_at'] = self.connection_started_at
            self.rows.append(row)
            if expired is not None:
                if self.gate is not None and self.gate.phase == 'stopping':
                    return  # Terminal brake replacement is the requested action.
                if self.gate is not None and self.gate.hold_resume:
                    self.gate.delay_hold(expired)
                    return
                self.fail(expired)
            if context is not None:
                context.setdefault('first_udp_send_started', now)
                context['last_udp_send_returned'] = row['udp_send_returned']
        except Exception as error:
            self.fail(error)


def install_checked_udp(api, gate=None):
    client = getattr(api, '_HexDeviceApi__kcp_client', None)
    if client is None:
        return None
    with client._kcp_lock:
        adapter = getattr(client, '_policy_udp', None)
        if adapter is None:
            adapter = CheckedUDP(client, gate)
        return adapter
