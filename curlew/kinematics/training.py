"""Coarse-to-fine warm-up of Fourier velocity amplitudes, longest wavelength first.

Mode ``k`` of wavelength ``ℓ_k`` sits at octave ``o_k = log2(L_max / ℓ_k)``,
and ``K = log2(L_max / L_min)`` over the modes that were drawn. Its amplitude
gradient is multiplied by

    w_k = ½ (1 − cos(π · clip(α − o_k + 1, 0, 1))),

with ``α`` rising from 0 to ``K`` over the first ``ramp`` fraction of epochs
(default 1/6). A closed window zeros that mode, and with it the phase and bias
that act only through the amplitude. A longer ramp lets the long modes commit
before the short ones open.

    with SpectralWarmup(event) as schedule:
        model.prefit(epochs=EPOCHS, schedule=schedule)
"""

from __future__ import annotations

import math

import torch

from curlew.fields.series import FSF

__all__ = ["SpectralWarmup"]


def _fsf_modules(target):
    """Every FSF under ``target``: a restore event, its field, a velocity, or a list of them."""
    if isinstance(target, (list, tuple)):
        return [m for t in target for m in _fsf_modules(t)]
    field = getattr(target, "field", None)
    if field is not None and hasattr(field, "_integrator"):
        target = field
    integrator = getattr(target, "_integrator", None)
    if integrator is not None:
        target = integrator.velocities
    if not isinstance(target, torch.nn.Module):
        raise TypeError(f"cannot find Fourier velocities in {type(target).__name__}")
    return list(FSF.iter_in(target))


class SpectralWarmup:
    """
    ``schedule(epoch, epochs)`` for :meth:`~curlew.fields.BaseSF.fit`.

    ``target`` is a restore event, its field, or the velocity module(s).
    ``ramp`` is the fraction of epochs over which the octaves open, in ``(0, 1]``.
    :attr:`n_octaves` is ``K``. As a context manager, or via :meth:`remove`,
    the gradient hooks come off when training ends.
    """

    def __init__(self, target, ramp: float = 1 / 6):
        modules = _fsf_modules(target)
        lengths = [2 * math.pi / m.omega_spatial.detach().norm(dim=1) for m in modules]
        if not lengths:
            raise ValueError("no Fourier modes found")
        l_max = float(max(float(l.max()) for l in lengths))
        l_min = float(min(float(l.min()) for l in lengths))
        self.n_octaves = math.log2(l_max / l_min)
        self.ramp = 1 / 6 if ramp is None else float(ramp)
        if not 0.0 < self.ramp <= 1.0:
            raise ValueError(f"ramp must be in (0, 1], got {ramp}")
        self._entries = []          # (octave per mode, window, hook handles)
        for m, ell in zip(modules, lengths):
            octave = torch.log2(l_max / ell).clamp(min=0.0)
            params = [getattr(m, n) for n in ("A_mu", "B_mu", "C_mu")
                      if isinstance(getattr(m, n, None), torch.Tensor) and getattr(m, n).requires_grad]
            if not params:
                continue
            window = torch.ones_like(octave)
            handles = [p.register_hook(lambda g, w=window: g * w.to(g.dtype).reshape(g.shape))
                       for p in params if p.shape == octave.shape]
            self._entries.append((octave, window, handles))
        # Every window starts open, so nothing changes until the fit loop calls the schedule.

    def update(self, alpha: float):
        """Set the windows for a given ``α`` (0: only the longest octave; ``K``: every mode)."""
        for octave, window, _ in self._entries:
            x = (alpha - octave + 1.0).clamp(0.0, 1.0)
            window.copy_(0.5 * (1.0 - torch.cos(math.pi * x)))

    def __call__(self, epoch: int, epochs: int):
        k = self.n_octaves
        self.update(k * min(1.0, epoch / (self.ramp * max(epochs, 1))))

    def remove(self):
        """Take the gradient hooks off (and open every mode)."""
        for _, window, handles in self._entries:
            for h in handles:
                h.remove()
            window.fill_(1.0)
        self._entries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False
