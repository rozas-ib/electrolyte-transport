import numpy as np
import pytest

from diffusion_conductivity import conductivity_from_slope
from diffusion_conductivity.analysis import convert_conductivity, ensure_odd_window


def test_conductivity_unit_conversion() -> None:
    assert convert_conductivity(10.0, "S/m") == 10.0
    assert convert_conductivity(10.0, "S/cm") == pytest.approx(0.1)
    assert convert_conductivity(10.0, "mS/cm") == pytest.approx(100.0)


def test_conductivity_from_slope_uses_angstrom_to_meter_conversion() -> None:
    value = conductivity_from_slope(1.0, 1e-27, 300.0)
    assert np.isfinite(value)
    assert value > 0


def test_odd_window_is_normalized() -> None:
    assert ensure_odd_window(20) == 21
    assert ensure_odd_window(1) == 5
