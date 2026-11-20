# Extend or edit the curlew codebase

**Goal:** Change curlew source, tests, or API — not build a geological model.

## Workflow

```
- [ ] Environment probe (scripts/check_env.py)
- [ ] Identify scope: bug fix, new field type, synthetic builder, test, docs
- [ ] Minimal diff; match existing conventions
- [ ] Run targeted pytest
```

## Conventions

- Follow patterns in neighbouring modules; reuse existing helpers.
- Keep code clear and minimal. Avoid writing private functions, where possible nest helper functions inside the main function being implemented (or implement inline).
- Provide one short explanatory comment per block of code (at least)
- Ensure each main function has a docstring, matching the style of the overall repository
- Prefer extending `curlew.geology` factories over ad-hoc `GeoEvent` construction in tests.
- Ensure relevant tests are run after making edits.

## Testing examples

```bash
python -m pytest tests/test_geology.py -k test_hutton -x
python -m pytest tests/test_analytic.py -x
python -m pytest tests/test_synthetic.py -x
```

Use the same `PYTHON:` from the environment probe for all commands.

## Geomodelling skills vs codebase work

Do **not** apply model-building playbooks when the user asks to refactor internals, fix imports, or add library features. Stop using [synthetic-model.md](synthetic-model.md) / [interpolated-model.md](interpolated-model.md) unless the change requires a new synthetic regression test.

For API behaviour, read docstrings and [api-reference.md](../api-reference.md); verify against tests before documenting new behaviour in skills.
