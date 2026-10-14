import math
import logging
from typing import TYPE_CHECKING, Dict

from omegaconf import OmegaConf, ListConfig

# Re-exported for backwards compatibility; the canonical home is `naming` so the
# core component/comm classes don't have to import OmegaConf. (#topic-registry)
from hexfellow_deploy.runtime.utils.naming import clean_name  # noqa: F401

if TYPE_CHECKING:
    from hexfellow_deploy.runtime.components.comms.base_comm import DABaseComm


def register_rl_resolvers():
    try:
        OmegaConf.register_new_resolver("eval", eval)
        OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
        OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
        OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
        OmegaConf.register_new_resolver("sum", lambda x: sum(x))
        OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
        OmegaConf.register_new_resolver("int", lambda x: int(x))
        OmegaConf.register_new_resolver("len", lambda x: len(x))
        OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
        OmegaConf.register_new_resolver("tuple", lambda x: tuple(x))
        OmegaConf.register_new_resolver(
            "neg",
            lambda x: [-1.0 * a for a in x] if isinstance(x, ListConfig) else -1.0 * x,
        )
    except Exception as e:
        logging.warning(f"Warning: Some resolvers already registered: {e}")


def build_topic_registry(topics_cfg) -> "Dict[str, DABaseComm]":
    """Instantiate every topic exactly once.

    Each entry in the top-level ``topics:`` block is a comm wrapper spec
    (``_target_`` + its shape params); the block key is the topic name. The name
    is injected into the wrapper (so specs never repeat it) and also keys the
    registry. The returned objects are the single owners of their shared-memory
    segments — every component that references a topic gets the same instance
    via the ``${topic:<name>}`` resolver.
    """
    from hydra.utils import instantiate

    registry: "Dict[str, DABaseComm]" = {}
    origin: "Dict[str, str]" = {}  # clean key -> original topic name (for error clarity)
    for name, spec in topics_cfg.items():
        key = clean_name(name)
        if key in registry:
            raise ValueError(
                f"Duplicate topic: '{name}' and '{origin[key]}' both map to "
                f"shared-memory name '{key}'."
            )
        origin[key] = name
        registry[key] = instantiate(spec, name=name)
    return registry


def register_topic_resolver(registry: "Dict[str, DABaseComm]") -> None:
    """Register ``${topic:<name>}`` to return the single shared comm for a topic.

    Every component config that references ``${topic:imu}`` resolves to the same
    object, so a topic is defined once (in ``topics:``) and referenced anywhere.
    """

    def _resolve(name: str) -> "DABaseComm":
        key = clean_name(name)
        if key not in registry:
            raise KeyError(
                f"Unknown topic '{name}'. Define it in the top-level 'topics:' block. "
                f"Known topics: {sorted(registry)}"
            )
        return registry[key]

    OmegaConf.register_new_resolver("topic", _resolve, replace=True, use_cache=False)
