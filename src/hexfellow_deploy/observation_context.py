"""Measured observation freshness and skill continuation helpers."""
import math


def wrist_context(topic, head_timestamp, now, max_age=.5, max_skew=.15, name="wrist"):
    """Do not send stale context or pretend the two cameras are synchronized."""
    from hexfellow_deploy.interfaces.contract import validate_rgb_camera
    frame = topic.get()
    if frame is None:
        return None
    validate_rgb_camera(frame)
    if not all(math.isfinite(x) for x in (head_timestamp,now)):
        raise ValueError('Invalid observation clock')
    age = now-frame['received_s']
    if not 0 <= age <= max_age or abs(frame['received_s']-head_timestamp) > max_skew:
        return None
    return {name:frame}


def checkpoint_observation_ready(feedback, head_timestamp, cameras):
    """All visual evidence for a continuation must postdate the checkpoint."""
    pending = feedback.get('execution', {}).get('skill', {})
    if pending.get('status') != 'awaiting_verification':
        return True
    after = pending.get('observation_after_s')
    return (isinstance(after, (int,float)) and math.isfinite(after)
            and head_timestamp > after and all(c['received_s'] > after for c in cameras.values()))


def post_action_hold(diagnostic):
    return ('observation_hold' if diagnostic.get('needs_observation') is True
            and diagnostic.get('skill', {}).get('status') == 'awaiting_verification' else 'hold')
