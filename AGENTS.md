# AGENTS.md

Guidance for AI coding agents working in this repository.

## Skills

Reusable agent instructions live in [`.agents/skills/`](.agents/skills).

Each skill is a directory containing a `SKILL.md` with YAML frontmatter (`name`,
`description`) followed by Markdown instructions. Skills may bundle supporting
references, task playbooks, and helper scripts alongside `SKILL.md`.

Currently available:

- [`curlew-geomodelling`](.agents/skills/curlew-geomodelling/SKILL.md) — building
  implicit geological models with curlew (`GeoModel`, `GeoEvent`, neural and
  analytical fields, faults, unconformities, dykes, folds), including Learnable
  Diffeomorphic Interpolation (`restore`, Euler velocities, shear composition);
  extracting geological constraints from maps and reports; visualising and
  validating models; extending the codebase. Start at `SKILL.md`; task-specific
  playbooks are in `tasks/`, with deeper references in `context.md`,
  `structures.md`, `restoration.md`, `api-reference.md`,
  `geological-reasoning.md`, and `examples.md`.

Load the relevant `SKILL.md` before starting a task it covers, and follow any
linked playbook rather than improvising. Before running curlew or PyTorch code,
follow the environment check in the skill's `environment.md`.

## Repository layout

| Path | Contents |
|---|---|
| `curlew/` | Package source |
| `tests/` | Test suite |
| `benchmarks/` | Benchmark scripts and results |
| `sandbox/` | Scratch notebooks and working data (git-ignored) |

## Conventions

- Python; `numpy`, `pytorch`, and `tqdm` are required, with `matplotlib`, `plyfile`, and `napari` optional. See `setup.py`.
- Everything is implemented differentiably in PyTorch.
- Run the test suite with `pytest` from the repository root.
