"""Dependency-free naming helpers.

Kept separate from :mod:`hexfellow_deploy.runtime.utils.config_utils` (which pulls in
OmegaConf/Hydra) so the core component and comm classes stay importable with
only numpy — e.g. from an external process attaching to a topic's shared memory.
"""


def clean_name(name: str) -> str:
    """Normalize a topic/component name into a shared-memory-safe identifier."""
    return name.replace("/", "__")
