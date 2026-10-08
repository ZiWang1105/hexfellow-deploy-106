"""Binary gripper request contract shared by policy adapters and motor control."""

def validate_gripper(value, *, allow_hold=False):
    if value is None and allow_hold:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1):
        raise ValueError('Gripper must be binary: 0 = close, 1 = open')
    return int(value)
