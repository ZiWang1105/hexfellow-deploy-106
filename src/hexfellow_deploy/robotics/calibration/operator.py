"""Operator-paced calibration rehearsal state machine; no hardware access.

Time refers to the recorded physics trace. Pausing freezes playback; it is not
a model of the deployed controller's braking response.
"""
from __future__ import annotations


class Rehearsal:
    def __init__(self, dwells, duration, *, heartbeat_timeout=2.0):
        self.dwells = dwells
        self.duration = float(duration)
        self.timeout = heartbeat_timeout
        self.time = 0.0
        self.destination = None
        self.pose = 0
        self.phase = 'ready'
        self.version = 0
        self.captured = set()
        self.message = 'Next pose starts the recorded approach to pose 1.'
        self.heartbeat_at = float('-inf')

    def heartbeat(self, now):
        self.heartbeat_at = now

    def tick(self, dt, now):
        if self.phase in ('moving', 'returning'):
            if now - self.heartbeat_at > self.timeout:
                self.command('pause', now=now)
                self.message = 'Browser heartbeat lost. Playback paused; Resume is required.'
                return
            self.time = min(self.destination, self.time + max(0, min(dt, 0.1)))
            if self.time >= self.destination:
                self.phase = 'complete' if self.destination == self.duration else 'at_pose'
                self.version += 1
                self.message = ('Returned to recorded start.' if self.phase == 'complete'
                                else f'Pose {self.pose}: recorded settling interval completed. Capture is available.')

    def command(self, action, *, now, version=None):
        # Pause must always work, including with an old browser state.
        if action != 'pause' and version is not None and version != self.version:
            raise ValueError('State changed. Refresh the status before issuing another action.')
        if action == 'pause':
            if self.phase in ('moving', 'returning'):
                self.phase = 'paused_return' if self.phase == 'returning' else 'paused'
                self.version += 1
                self.message = 'Playback paused. Capture stays disabled between poses.'
            return
        if action in ('next', 'resume', 'return') and now - self.heartbeat_at > self.timeout:
            raise ValueError('A fresh browser heartbeat is required.')
        if action == 'next':
            if self.phase not in ('ready', 'at_pose'):
                raise ValueError('Finish or resume the current movement first.')
            if self.pose and self.pose not in self.captured:
                raise ValueError('Capture this pose before proceeding.')
            if self.pose >= len(self.dwells):
                raise ValueError('All poses visited. Solve or return to the recorded start.')
            self.destination = float(self.dwells[self.pose]['time_s'])
            self.pose += 1
            self.phase = 'moving'
            self.message = f'Playing the checked route and settling interval to pose {self.pose}.'
        elif action == 'resume':
            if self.phase not in ('paused', 'paused_return'):
                raise ValueError('Playback is not paused.')
            self.phase = 'returning' if self.phase == 'paused_return' else 'moving'
            self.message = 'Resumed explicitly by the operator.'
        elif action == 'capture':
            if self.phase != 'at_pose':
                raise ValueError('Capture requires arrival at a recorded stationary pose.')
            # Disk I/O/detection validation must succeed before calling this.
            self.captured.add(self.pose)
            self.message = f'Pose {self.pose} saved. Next pose is a separate action.'
        elif action == 'return':
            if self.phase != 'at_pose' or len(self.captured) != len(self.dwells):
                raise ValueError('Return playback is available after all poses are captured.')
            self.destination = self.duration
            self.phase = 'returning'
            self.message = 'Playing the validated reverse route.'
        else:
            raise ValueError(f'Unknown action: {action}')
        self.version += 1

    def status(self):
        return dict(mode='simulation_rehearsal', phase=self.phase, version=self.version,
                    pose=self.pose, count=len(self.dwells), captured=sorted(self.captured),
                    time_s=self.time, message=self.message,
                    can_next=self.phase in ('ready', 'at_pose') and
                    (not self.pose or self.pose in self.captured) and self.pose < len(self.dwells),
                    can_capture=self.phase == 'at_pose',
                    can_pause=self.phase in ('moving', 'returning'),
                    can_resume=self.phase in ('paused', 'paused_return'),
                    can_solve=len(self.captured) == len(self.dwells) and
                    self.phase in ('at_pose', 'complete'),
                    can_return=self.phase == 'at_pose' and len(self.captured) == len(self.dwells))
