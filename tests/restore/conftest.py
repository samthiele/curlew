"""Restore-event tests.

The suite follows the object that ``restore()`` builds, from the velocity
out to the event:

* ``test_euler.py`` — Euler potentials and the divergence-free velocity
* ``test_flow.py`` — RK2 / RK4 and the injectivity margin
* ``test_shear.py`` — exact shear integrator
* ``test_normal_split.py`` — full 3-D normal grouping of that shear
* ``test_composite.py`` — banded composition of several velocities
* ``test_pin.py`` — restored-frame translation, applied after fitting
* ``test_restoration.py`` — ``RestorationField`` losses and the ``restore()`` event
* ``test_lift.py`` — fault labels carried through the flow
* ``test_band.py`` — wavelength band read from the constraints
* ``test_training.py`` — spectral warm-up of that band

Inequality margins, deferred loss weights, the Fourier series, and meshing
stay under ``tests/``: the event uses them, and so does the rest of the package.
"""
import curlew
import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_defaults():
    """Pin CPU float64 for this test, then put the process globals back.

    ``curlew.device``, ``curlew.dtype`` and ``curlew.default_dim`` are process
    globals. A 3-D case in this folder must not leak into the next file.
    """
    device, dtype, dim = curlew.device, curlew.dtype, curlew.default_dim
    curlew.device = "cpu"
    curlew.dtype = torch.float64
    yield
    curlew.device, curlew.dtype, curlew.default_dim = device, dtype, dim
