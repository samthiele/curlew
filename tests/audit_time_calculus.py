"""
Numerical audit: does the time-varying TemporalFSF velocity break any of the
restoration calculus? Checks, all in float64:

1. RK2 convergence order for the non-autonomous forward map (should be ~2).
2. inverse(forward(x)) -> identity at O(dt^2).
3. Magnus-accumulated Jacobian vs autograd Jacobian of the integrated map.
4. det J == 1 (exactly, via expm_traceless) at every step count.
5. Nanson normals vs autograd gradient of the restored depth scalar.
6. Divergence-free at interior times.
7. Smoothness of v in t (no kinks from _clamp_t inside (0,1)).
"""
import numpy as np
import torch

import curlew

curlew.default_dim = 2
curlew.device = "cpu"
curlew.dtype = torch.float64

from curlew.core import HSet
from curlew.fields.clebsch import temporal_clebsch_velocity
from curlew.geology.interactions import FlowOffset

torch.manual_seed(0)

vel = temporal_clebsch_velocity(
    2,
    n_features=64,
    length_scale_range=(0.4, 3.0),
    freq_sampling="quasi",
    activation="cubic",
    seed=7,
    init_std=0.0,
    anchor=[0.0, 0.0],
    killing="translation",
)
vel.beta.set_amplitudes(torch.abs(torch.randn_like(vel.beta.A_rho)) * 0.15)

x = torch.randn(40, 2, dtype=curlew.dtype) * 0.8


def integrate(n, sign=+1.0, jac=False, xx=None):
    f = FlowOffset(vel, n_steps=n)
    return f._integrate(x if xx is None else xx, sign=sign, jac=jac)


print("== 1. RK2 convergence order (non-autonomous forward map) ==")
with torch.no_grad():
    ref = integrate(1024)
    errs = []
    for n in (8, 16, 32, 64):
        e = (integrate(n) - ref).norm(dim=-1).max().item()
        errs.append(e)
        print(f"  n={n:3d}  max err = {e:.3e}")
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    print(f"  observed orders: {[f'{o:.2f}' for o in orders]}  (expect ~2)")

print("\n== 2. inverse(forward(x)) -> identity ==")
with torch.no_grad():
    for n in (8, 16, 32, 64):
        fwd = integrate(n, sign=+1.0)
        back = integrate(n, sign=-1.0, xx=fwd)
        e = (back - x).norm(dim=-1).max().item()
        print(f"  n={n:3d}  roundtrip err = {e:.3e}")

print("\n== 3. Magnus Jacobian vs autograd Jacobian of the map ==")
n = 64
f = FlowOffset(vel, n_steps=n)
xg = x[:8].clone().requires_grad_(True)
x_end, J_mag = f._integrate(xg, sign=-1.0, jac=True)
rows = []
for i in range(2):
    g = torch.autograd.grad(x_end[:, i].sum(), xg, retain_graph=True)[0]
    rows.append(g)
J_auto = torch.stack(rows, dim=1)
diff = (J_mag - J_auto).abs().max().item()
print(f"  n={n}: max |J_magnus - J_autograd| = {diff:.3e}")
print(f"  (autograd differentiates the RK2 scheme itself; Magnus is a separate")
print(f"   2nd-order update, so agreement should be O(dt^2), not machine eps)")
f2 = FlowOffset(vel, n_steps=2 * n)
xg2 = x[:8].clone().requires_grad_(True)
x_end2, J_mag2 = f2._integrate(xg2, sign=-1.0, jac=True)
rows2 = []
for i in range(2):
    g = torch.autograd.grad(x_end2[:, i].sum(), xg2, retain_graph=True)[0]
    rows2.append(g)
J_auto2 = torch.stack(rows2, dim=1)
diff2 = (J_mag2 - J_auto2).abs().max().item()
print(f"  n={2*n}: max diff = {diff2:.3e}  (ratio {diff/max(diff2,1e-300):.1f}, expect ~4)")

print("\n== 4. det J == 1 (volume preservation, time-varying) ==")
with torch.no_grad():
    for n in (4, 16, 64):
        _, J = integrate(n, sign=-1.0, jac=True)
        d = (torch.linalg.det(J) - 1.0).abs().max().item()
        print(f"  n={n:3d}  max |det J - 1| = {d:.3e}")

print("\n== 5. Nanson normals vs autograd grad of restored depth ==")
from curlew.fields.restoration import RestorationField

field = RestorationField(
    "audit", H=HSet(), input_dim=2,
    velocity=vel, n_steps=32, depth_axis=1, signed_normals=True,
)
xq = x[:10].clone().requires_grad_(True)
depth = field.integrator._integrate(xq, sign=-1.0)[:, 1]
g_auto = torch.autograd.grad(depth.sum(), xq)[0]
g_auto = torch.nn.functional.normalize(g_auto, dim=-1)
with torch.no_grad():
    n_pred = field.predict_normals(x[:10])
mismatch = (n_pred - g_auto).abs().max().item()
print(f"  max |nanson - autograd_grad(phi)| = {mismatch:.3e}")

print("\n== 6. div v == 0 at interior t ==")
with torch.no_grad():
    for t in (0.0, 0.13, 0.5, 0.87, 1.0):
        _, jv = vel.forward_and_jacobian(x, t=t)
        div = (jv[:, 0, 0] + jv[:, 1, 1]).abs().max().item()
        print(f"  t={t:.2f}  max|div v| = {div:.3e}")

print("\n== 7. smoothness of v(x,t) in t (finite-difference dv/dt) ==")
with torch.no_grad():
    ts = np.linspace(0.001, 0.999, 200)
    xprobe = x[:5]
    vs = torch.stack([vel(xprobe, t=float(t)) for t in ts])
    dv = (vs[1:] - vs[:-1]).norm(dim=-1).max(dim=-1).values / (ts[1] - ts[0])
    ratios = (dv[1:] / dv[:-1].clamp_min(1e-12))
    print(f"  max |dv/dt| = {dv.max().item():.3e}, "
          f"max step-to-step ratio = {ratios.max().item():.3f} (near 1 = smooth)")

print("\nAll audits complete.")
