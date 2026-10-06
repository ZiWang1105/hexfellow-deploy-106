"""Operator-started sequence decisions; no robot I/O or synthetic live heartbeat."""
class Sequence:
    def __init__(self, *, pose_count=30, preparation='prepare', return_before_solve=False,
                 return_after_solve=False):
        self.pose_count=pose_count
        self.preparation=preparation
        self.return_before_solve=return_before_solve
        self.return_after_solve=return_after_solve
        self.mode='paused'
        self.interruption=0
        self.stable_since=None
        self.stable_pose=None
        self.preparation_requested=False

    def start(self, interruption=0, returning=False):
        self.mode='returning' if returning else 'running'
        self.interruption=interruption
        self.stable_since=None
        self.preparation_requested=False

    def pause(self):
        self.mode='paused'
        self.stable_since=None

    def step(self, *, now, prepared, phase, arrived, target, captured, solved,
             interrupted, paused, busy, fresh_operator, ineligible=(), pending_invalid=None):
        if self.mode not in ('running','returning'):return None
        if interrupted!=self.interruption or not fresh_operator:
            self.pause();return None
        if busy:return None
        if not prepared:
            if self.preparation_requested:return None
            self.preparation_requested=True
            return self.preparation
        if phase not in ('hold','ready'):
            self.stable_since=None;return None
        if self.mode=='returning':
            if arrived==target==-1 and not paused:
                self.mode='paused';return None
            return 'return'
        if pending_invalid is not None:
            return 'skip-invalid' if pending_invalid['ready'] else None
        if paused:return 'resume'
        attempted=set(captured)|set(ineligible)
        if len(attempted)==self.pose_count:
            if self.return_before_solve and not (arrived==target==-1):return 'return'
            if not solved:return 'solve'
            if self.return_after_solve and not (arrived==target==-1):return 'return'
            self.mode='complete';return None
        if arrived>0 and arrived==target and arrived not in attempted:
            if self.stable_pose!=arrived or self.stable_since is None:
                self.stable_pose=arrived;self.stable_since=now
            if now-self.stable_since>=1.:return 'capture'
            return None
        self.stable_since=None
        return 'next'
