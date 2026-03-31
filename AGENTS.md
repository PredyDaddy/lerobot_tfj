# Repository Guidelines

## Project Structure & Module Organization
Primary library code lives in `src/lerobot/`, organized by domain (`policies/`, `datasets/`, `robots/`, `envs/`, `cameras/`, `async_inference/`, `rl/`). Tests mirror that layout under `tests/`. Use `examples/` for runnable samples, `docs/source/` for user docs, and `scripts/` for repo-specific utilities. Treat `tfj_envs/`, `web/`, `transformers/`, `VITA/`, `streaming-flow-policy/`, and `fpo/` as separate or experimental areas unless your change explicitly targets them.

## Build, Test, and Development Commands
Set up a dev environment with either:

```bash
uv sync --extra dev --extra test
pip install -e ".[dev,test]"
```

Use `ruff check src tests` to lint, `pytest tests -q` for the main suite, and `LEROBOT_TEST_DEVICE=cpu pytest tests/policies/test_policies.py -q` for targeted CPU-safe runs. Run `make test-end-to-end DEVICE=cpu` for the lightweight train/eval smoke workflow defined in the root `Makefile`.

## Coding Style & Naming Conventions
Target Python 3.10+, use 4-space indentation, and keep code compatible with the Ruff rules in `pyproject.toml`. The configured line length is 110. Prefer `snake_case` for modules, functions, and test files; use `PascalCase` for classes; keep policy, robot, and environment names aligned with the registries exposed by `lerobot`. Favor small, typed helpers over large ad hoc scripts.

## Testing Guidelines
Write tests with `pytest` and place them in the matching domain folder, for example `tests/datasets/test_<feature>.py`. Reuse shared fixtures from `tests/fixtures/` and plugins from `tests/plugins/`. Hardware- or extra-dependent tests should skip cleanly when dependencies are unavailable. No explicit coverage threshold is declared here, so every behavior change should include focused regression tests.

## Commit & Pull Request Guidelines
Recent commits favor short imperative subjects, often with prefixes such as `chore:`; follow that pattern when possible (`feat:`, `fix:`, `chore:`). Keep PRs scoped, describe motivation and impact, list validation commands you ran, and link the relevant issue or task. Include screenshots or logs for UI, docs, training, or robot-runtime changes.

## Artifacts & Local Data
Do not commit local outputs or generated state from `outputs/`, `logs/`, `build/`, `.venvs/`, `tmp*/`, or dataset snapshots unless the change is intentionally about those artifacts.


- use multi-agent mode to analysis any problem and reply any problem 
- the multi-agnet must be gpt-5.4 xhigh
