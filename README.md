# Diffusion Conductivity

Diffusion Conductivity is a Python library and command-line tool for vectorized
Helfand--Einstein transport analysis of molecular-dynamics trajectories. It
calculates ionic conductivity using multi-time-origin averaging, with total,
self, distinct, species, and cross contributions.

The package supports atomic and centre-of-mass species, charge restoration,
optional PBC transformations, multiple replicas, Onsager/diffusion-like
diagnostics, and binary-electrolyte transference-number estimates.

## Example system

The included [LiFSI in DME:TOL configuration](examples/lifsi_dme_tol.toml) is
an editable workflow example. Validate selections, charge restoration, PBC
treatment, and fit windows for every system before interpreting results.
Topology and trajectory data are intentionally not included in the repository.

## Installation

```bash
python -m pip install .
```

For tests:

```bash
python -m pip install ".[test]"
```

## Command-line use

Copy and adapt the example configuration, then run:

```bash
diffusion-conductivity my_analysis.toml
```

To create a commented configuration template:

```bash
diffusion-conductivity --write-template analysis.toml
```

The analysis writes a timestamped execution log, averaged time-series CSV,
conductivity diagnostic PNG, and (when enabled) replica CSVs and an
Onsager/diffusion diagnostic PNG. File names start with the configured output
prefix.

## Python API

```python
from diffusion_conductivity import run_conductivity_analysis

summary = run_conductivity_analysis("my_analysis.toml")
print(summary["conductivity_s_m"])
```

## Development

```bash
pytest
```

Before public release, add a license chosen by the copyright holder and a
`CITATION.cff` file for the associated publication.
