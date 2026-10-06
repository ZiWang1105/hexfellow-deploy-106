"""Calibration overlay; control, assets and robot definitions remain upstream.

Only calibration is owned here. Extending the package search path avoids copying
or pinning an old version of the deployed robot controller.
"""
import os
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
_upstream = Path(os.environ.get('HEXFELLOW_CALIBRATION_UMI', _root.parent.parent/'UMI_Hardware'))
_dependency = _upstream/'src/policy_sim'
if not (_dependency/'contract.py').is_file():
    raise ImportError(f'Initialize the UMI_Hardware submodule: {_upstream}')
__path__ = [str(Path(__file__).parent), str(_dependency)]
from .contract import (ACTION_DIM, ACTION_HORIZON, CONTRACT_ID, OBSERVATION_KEYS,
    REPLAN_STRIDE_STEPS, STATE_DIM, build_policy_state, decode_anchor_relative_chunk,
    schema_hash, validate_action_chunk)
__all__ = ['ACTION_DIM', 'ACTION_HORIZON', 'CONTRACT_ID', 'OBSERVATION_KEYS',
    'REPLAN_STRIDE_STEPS', 'STATE_DIM', 'build_policy_state',
    'decode_anchor_relative_chunk', 'schema_hash', 'validate_action_chunk']
