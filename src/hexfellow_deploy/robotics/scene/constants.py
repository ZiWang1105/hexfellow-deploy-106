"""Shared geometry and semantic names for tabletop scenes."""

from __future__ import annotations

import numpy as np

TABLE_CENTER = np.array([0.75, 0.0, 0.695])
TABLE_TOP_Z = 0.72
TABLE_USABLE_RADIUS = 0.447
OBJECT_GAP = 0.006

ORGANIZE_GOAL_SITE = "organize_goal_volume"
ORGANIZE_BOX_OUTER_HALF_EXTENTS = (0.17, 0.14)
ORGANIZE_BOX_INNER_HALF_EXTENTS = (0.158, 0.128)
ORGANIZE_BOX_WALL_HEIGHT = 0.11

TRASHCAN_GOAL_SITE = "trashcan_goal_volume"
TRASHCAN_POSITION = np.array([0.60, -0.68, 0.001])
TRASHCAN_OUTER_RADIUS = 0.205
TRASHCAN_INNER_RADIUS = 0.180
TRASHCAN_HEIGHT = 0.45
