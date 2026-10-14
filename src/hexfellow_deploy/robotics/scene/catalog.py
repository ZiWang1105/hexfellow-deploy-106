"""Tabletop object catalogue and deterministic identity resolution."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..paths import ASSET_DIR

OBJECT_ASSET_DIR = ASSET_DIR / "objects"
FOOTPRINT_PADDING = 0.004
CATEGORY_ALIASES = ("plate", "bowl", "snack_box", "fruit")
RANDOM_OBJECT_CATEGORIES = (
    "lotion",
    "toy",
    "mug",
    "tape",
    "remote",
    "plate",
    "bowl",
    "snack_box",
    "apple",
    "orange",
    "banana",
    "lemon",
    "pear",
)


@dataclass(frozen=True)
class ObjectSpec:
    """Static catalog information used for assets, physics, and layout."""

    object_id: str
    mass: float
    size: tuple[float, float, float]
    directory: str | None = None
    category: str | None = None
    asset_group: str = "gso"
    collision_hulls: int = 32
    preferred_pos: tuple[float, float] | None = None
    preferred_yaw: float = 0.0

    @property
    def is_scanned(self) -> bool:
        return self.directory is not None

    @property
    def asset_directory(self) -> Path:
        if self.directory is None:
            raise ValueError(f"{self.object_id} has no scanned asset directory")
        return OBJECT_ASSET_DIR / self.asset_group / self.directory

    @property
    def footprint_radius(self) -> float:
        return 0.5 * max(self.size[:2]) + FOOTPRINT_PADDING


def _spec(
    object_id: str,
    mass: float,
    size: tuple[float, float, float],
    directory: str | None = None,
    **kwargs: object,
) -> ObjectSpec:
    return ObjectSpec(object_id, mass, size, directory, **kwargs)


OBJECT_CATALOG = {
    spec.object_id: spec
    for spec in (
        _spec(
            "lotion",
            0.25,
            (0.056661, 0.050695, 0.195286),
            "Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion",
            preferred_pos=(0.54, 0.30),
            preferred_yaw=math.radians(14),
        ),
        _spec(
            "toy",
            0.10,
            (0.066010, 0.045905, 0.080687),
            "Android_Figure_Orange",
            preferred_pos=(0.48, -0.27),
            preferred_yaw=math.radians(-20),
        ),
        _spec(
            "mug",
            0.34,
            (0.165662, 0.125540, 0.094769),
            "Cole_Hardware_Mug_Classic_Blue",
            preferred_pos=(0.98, 0.27),
            preferred_yaw=math.radians(35),
        ),
        _spec(
            "tape",
            0.10,
            (0.110786, 0.110415, 0.026684),
            "3M_Vinyl_Tape_Green_1_x_36_yd",
            preferred_pos=(1.07, -0.12),
            preferred_yaw=math.radians(10),
        ),
        _spec("remote", 0.12, (0.170, 0.052, 0.018), preferred_pos=(0.86, -0.32), preferred_yaw=math.radians(-12)),
        _spec("plate_green_small", 0.12, (0.0966, 0.0967, 0.0243), "Kotobuki_Saucer_Dragon_Fly", category="plate"),
        _spec(
            "plate_turquoise_medium",
            0.40,
            (0.2219, 0.2219, 0.0284),
            "Room_Essentials_Salad_Plate_Turquoise",
            category="plate",
        ),
        _spec(
            "plate_red_rim_large",
            0.70,
            (0.2745, 0.2740, 0.0333),
            "Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring",
            category="plate",
        ),
        _spec(
            "bowl_yellow_blue_small",
            0.18,
            (0.1135, 0.1149, 0.0619),
            "Cole_Hardware_Bowl_Scirocco_YellowBlue",
            category="bowl",
        ),
        _spec(
            "bowl_turquoise_medium", 0.35, (0.1569, 0.1572, 0.0826), "Room_Essentials_Bowl_Turquiose", category="bowl"
        ),
        _spec(
            "bowl_red_large",
            0.30,
            (0.2826, 0.2851, 0.1545),
            "Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl",
            category="bowl",
        ),
        _spec(
            "snack_box_nips",
            0.12,
            (0.0885, 0.0424, 0.1533),
            "Nestle_Nips_Hard_Candy_Peanut_Butter",
            category="snack_box",
        ),
        _spec(
            "snack_box_crunch",
            0.23,
            (0.1386, 0.0632, 0.1640),
            "Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box",
            category="snack_box",
        ),
        _spec(
            "apple",
            0.18,
            (0.085, 0.085, 0.090),
            "apple",
            category="fruit",
            asset_group="kenney_food_kit",
            collision_hulls=1,
        ),
        _spec(
            "orange",
            0.16,
            (0.080, 0.080, 0.080),
            "orange",
            category="fruit",
            asset_group="kenney_food_kit",
            collision_hulls=1,
        ),
        _spec(
            "banana",
            0.12,
            (0.050, 0.200, 0.060),
            "banana",
            category="fruit",
            asset_group="kenney_food_kit",
            collision_hulls=1,
        ),
        _spec(
            "lemon",
            0.12,
            (0.075, 0.085, 0.100),
            "lemon",
            category="fruit",
            asset_group="kenney_food_kit",
            collision_hulls=1,
        ),
        _spec(
            "pear",
            0.18,
            (0.075, 0.075, 0.105),
            "pear",
            category="fruit",
            asset_group="kenney_food_kit",
            collision_hulls=1,
        ),
        _spec(
            "apple_scan",
            0.18,
            (0.085, 0.085, 0.090),
            "apple",
            category="fruit",
            asset_group="sketchfab",
            collision_hulls=1,
        ),
        _spec(
            "orange_scan",
            0.16,
            (0.080, 0.080, 0.080),
            "orange",
            category="fruit",
            asset_group="sketchfab",
            collision_hulls=1,
        ),
    )
}


def _object_is_available(object_id: str) -> bool:
    spec = OBJECT_CATALOG[object_id]
    return not spec.is_scanned or (spec.asset_directory / "model.xml").exists()


def print_object_catalog() -> None:
    """Print installed exact IDs and deterministic category aliases."""
    print("Exact object IDs:")
    for spec in OBJECT_CATALOG.values():
        category = spec.category or "object"
        dimensions = "x".join(f"{value * 100:.1f}" for value in spec.size)
        availability = ""
        if not _object_is_available(spec.object_id):
            availability = "  [optional import required]" if spec.asset_group == "sketchfab" else "  [missing]"
        print(f"  {spec.object_id:30s} {category:9s} {dimensions} cm  {spec.mass:.2f} kg{availability}")
    print("\nSeeded category aliases:\n  plate\n  bowl\n  snack_box\n  fruit\n\nSpecial selections:\n  random\n  none")


def normalize_object_tokens(requested: Sequence[str] | str | None) -> list[str] | None:
    if requested is None:
        return None
    raw_tokens = [requested] if isinstance(requested, str) else list(requested)
    return [part.strip() for token in raw_tokens for part in token.split(",") if part.strip()]


def resolve_object_ids(requested: Sequence[str] | str | None, layout_seed: int = 0) -> tuple[str, ...]:
    """Resolve exact IDs and seeded category aliases into unique exact IDs."""
    if layout_seed < 0:
        raise ValueError("layout_seed must be non-negative")
    tokens = normalize_object_tokens(requested)
    selection_rng = np.random.default_rng(np.random.SeedSequence([layout_seed, 0x53454C]))
    random_selection = tokens is None or tokens == ["random"]
    if random_selection:
        available = [
            token
            for token in RANDOM_OBJECT_CATEGORIES
            if token in CATEGORY_ALIASES or (token in OBJECT_CATALOG and _object_is_available(token))
        ]
        count = int(selection_rng.integers(3, min(6, len(available)) + 1))
        tokens = [str(value) for value in selection_rng.choice(available, count, replace=False)]
    if not tokens:
        return ()
    if "random" in tokens:
        raise ValueError("'random' cannot be combined with other object IDs")
    if "none" in tokens:
        if len(tokens) != 1:
            raise ValueError("'none' cannot be combined with other object IDs")
        return ()
    invalid = sorted(set(tokens) - (set(OBJECT_CATALOG) | set(CATEGORY_ALIASES)))
    if invalid:
        raise ValueError(f"Unknown object selection: {', '.join(invalid)}. Run list-objects for valid IDs.")

    explicit = {token for token in tokens if token in OBJECT_CATALOG}
    selected: list[str] = []
    seen_aliases: set[str] = set()
    for token in tokens:
        if token in OBJECT_CATALOG:
            if token not in selected:
                selected.append(token)
            continue
        if token in seen_aliases:
            continue
        seen_aliases.add(token)
        candidates = [
            spec.object_id
            for spec in OBJECT_CATALOG.values()
            if spec.category == token
            and _object_is_available(spec.object_id)
            and (not random_selection or max(spec.size[:2]) <= 0.235)
            and spec.object_id not in explicit
            and spec.object_id not in selected
        ]
        if not candidates:
            candidates = [
                spec.object_id
                for spec in OBJECT_CATALOG.values()
                if spec.category == token
                and _object_is_available(spec.object_id)
                and (not random_selection or max(spec.size[:2]) <= 0.235)
            ]
        if not candidates:
            raise ValueError(f"No installed assets are available for category '{token}'")
        resolved = str(selection_rng.choice(candidates))
        if resolved not in selected:
            selected.append(resolved)
    return tuple(selected)
