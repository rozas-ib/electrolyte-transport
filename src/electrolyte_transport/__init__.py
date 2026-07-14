"""Vectorized Helfand--Einstein transport analysis for MD trajectories."""

from .analysis import (
    conductivity_from_slope,
    get_replica_specs,
    load_toml,
    run_conductivity_analysis,
    write_template_config,
)

__all__ = [
    "conductivity_from_slope",
    "get_replica_specs",
    "load_toml",
    "run_conductivity_analysis",
    "write_template_config",
]

__version__ = "0.1.0"
