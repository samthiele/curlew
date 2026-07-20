"""Tests for fault wiring in :func:`~curlew.geology.restore`."""
import curlew
import pytest
import torch

from curlew.core import CSet, HSet
from curlew.fields.clebsch import clebsch_velocity
from curlew.fields.lift import FaultLift
from curlew.geology import restore


@pytest.fixture(autouse=True)
def _restore_defaults():
    curlew.default_dim = 2
    curlew.device = "cpu"
    curlew.dtype = torch.float64


def test_restore_wires_fault_lift_from_traces():
    trace = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    fault_lift = FaultLift([trace])
    velocity = clebsch_velocity(2, n_features=8, lift_dim=fault_lift.lift_dim)
    event = restore(
        "fold", C=CSet(), H=HSet(), velocity=velocity, fault_lift=fault_lift,
    )
    assert event.field.fault_lift is fault_lift
    assert event.field.fault_lift.dim == 2
    assert event.field.integrator.fault_lift is fault_lift
    assert event.field.velocity.beta.lift_dim == fault_lift.lift_dim


def test_restore_wires_fault_lift_from_meshes():
    curlew.default_dim = 3
    verts = torch.tensor(
        [[1.0, 0.0, -10.0], [1.0, 2.0, -10.0], [1.0, 2.0, 10.0], [1.0, 0.0, 10.0]]
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    fault_lift = FaultLift([(verts, faces)])
    velocity = clebsch_velocity(3, n_features=8, lift_dim=fault_lift.lift_dim)
    event = restore(
        "fold3d",
        C=CSet(),
        H=HSet(),
        velocity=velocity,
        fault_lift=fault_lift,
        input_dim=3,
    )
    assert event.field.fault_lift is fault_lift
    assert event.field.fault_lift.dim == 3
    assert event.field.velocity.beta.lift_dim == fault_lift.lift_dim
