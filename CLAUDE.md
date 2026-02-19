# CLAUDE.md — uv Python project

## Project management with uv

This project uses [uv](https://docs.astral.sh/uv/) for all Python tooling.

### Hard rules

- **Never** use `pip`, `pip install`, `python -m venv`, or run `python` directly
- **Always** prefix every Python/tool invocation with `uv run`
- **Never** hand-edit `[project.dependencies]` or `[dependency-groups]` in pyproject.toml — use `uv add` / `uv remove` which also updates `uv.lock`
- **Always** commit `uv.lock` to git
- **Never** commit or manually create `.venv/`
- **Never** use pyenv, asdf, conda, or system package managers for Python — uv manages Python installations via `uv python install`

### Running things

- `uv run` is the single entry point — it creates `.venv`, installs the pinned Python from `.python-version`, syncs deps from `uv.lock`, installs the project editably (if a build system is defined), then runs the command
- Use `uv run --with <pkg>` for ephemeral dependencies not added to the project
- Use `uv run --no-project` to bypass project installation when running standalone scripts inside a project directory

### Dependencies

- `uv add <pkg>` — add a runtime dependency
- `uv add --dev <pkg>` — add to the `dev` dependency group (PEP 735)
- `uv add --group <name> <pkg>` — add to a named dependency group
- `uv add` supports version constraints, extras, git URLs (`git+https://...`), `--tag`/`--branch`/`--rev`, local paths (`--editable`), and bulk import (`-r requirements.txt`)
- `uv remove <pkg>` — remove a dependency
- `uv lock --upgrade-package <pkg>` — update a specific package to latest compatible version

### Environment and sync

- `uv sync` makes the environment match `uv.lock` exactly (removes extraneous packages)
- Default behavior includes dev dependencies; use `--no-dev` for production, `--group <name>` for named groups, `--all-groups` for everything
- `--no-install-project` installs only dependencies without the project itself
- `--frozen` skips lockfile checks (fast CI); `--locked` errors if lockfile is outdated (strict CI)
- `uv lock --check` verifies the lockfile is up-to-date without modifying it

### Python versions

- `uv python install <version>` — install a specific version (supports CPython, PyPy via `pypy@<version>`, GraalPy)
- `uv python pin <version>` — write `.python-version`
- `uv python list` — show available/installed versions
- `uv python upgrade <version>` — upgrade to latest patch release
- `--python <version>` on any command overrides the version for that invocation
- uv downloads Python automatically when needed — no manual installation required

### Standalone scripts (PEP 723)

- `uv init --script <file.py>` creates a script with inline metadata block (`# /// script ... # ///`)
- `uv add --script <file.py> <pkg>` adds dependencies to the script's metadata
- `uv lock --script <file.py>` creates a lockfile for reproducibility
- Scripts with a `#!/usr/bin/env -S uv run --script` shebang can be run directly after `chmod +x`

### Tools (uvx)

- `uvx <tool>` runs a CLI tool in an isolated temporary environment without installing it (equivalent to `uv tool run`)
- `uvx <tool>@<version>` pins a specific version; `uvx --from '<pkg>[extras]' <cmd>` for extras or when the command name differs from the package
- `uv tool install <tool>` installs globally; `uv tool upgrade <tool>` / `uv tool upgrade --all` to update
- **Use `uvx`** for standalone tools that don't need your project (formatters, linters on arbitrary code)
- **Use `uv run`** for tools that need your project importable (pytest, mypy checking your code)

### Building and distributing

- `uv build` creates sdist and wheel in `dist/`; `--wheel` or `--sdist` for one format only
- `uv build --no-sources` verifies the package builds without uv-specific sources — run before publishing
- Build backend is defined in `[build-system]` in pyproject.toml; supported: `uv_build` (default), `hatchling`, `flit-core`, `pdm-backend`, `setuptools`, `maturin`, `scikit-build-core`
- Add `classifiers = ["Private :: Do Not Upload"]` to prevent accidental PyPI publication

### Exporting

- `uv export --format requirements.txt` exports the lockfile to pip-compatible format
- `--output-file <path>` writes to a file; `--no-emit-project` omits the project itself (useful for Docker)
- `uv export --format pylock.toml` exports to PEP 751 format

### Creating new projects

- `uv init <name>` — application (flat layout, no build system)
- `uv init --package <name>` — packaged application (src layout + build system)
- `uv init --lib <name>` — library (src layout, always packaged)
- `--build-backend <backend>` selects the build backend; `--script <file.py>` creates a PEP 723 script

### Workspaces

- Define members in root pyproject.toml under `[tool.uv.workspace]` with `members` glob patterns
- All members share a single `uv.lock`; use `--package <name>` to target a specific member
- Workspace dependencies use `{ workspace = true }` in `[tool.uv.sources]` and are editable by default

### Jupyter

- `uv run --with jupyter jupyter lab` launches Jupyter as an ephemeral dependency
- For persistent kernels: `uv add --dev ipykernel`, then install a named kernel via `uv run ipython kernel install --user --name=<name>`
- In notebooks: `!uv add <pkg>` persists to pyproject.toml; `!uv pip install <pkg>` is session-only
- VS Code notebooks require `ipykernel` in the project environment — select the `.venv` Python as kernel

### PyTorch

- PyTorch publishes separate builds per accelerator (CPU, CUDA, ROCm, XPU) on dedicated indexes
- Configure via `[[tool.uv.index]]` with `explicit = true` (restricts the index to only packages listed in `[tool.uv.sources]`) and `[tool.uv.sources]` entries for `torch`/`torchvision`
- Use environment markers (`sys_platform`, `python_version`) for platform-specific index selection — PyTorch has no CUDA builds for macOS
- Available index suffixes: `cpu`, `cu118`, `cu126`, `cu128`, `cu130`, `rocm6.4`, `xpu`

---

## Python best practices

### Declarative and functional over imperative

- Express transformations as comprehensions, `map`/`filter`, generator expressions, and `sum`/`min`/`max`/`any`/`all` — not as manual loops that accumulate into mutable containers
- Use loops only when side effects are the primary purpose (I/O, mutation of external state)
- Prefer single-expression solutions over multi-step accumulation — if the result can be expressed as one comprehension or built-in call, do that

### Monadic error handling

- Return `T | None` instead of raising exceptions for expected failure modes (item not found, parse failure, optional lookup)
- For richer error context, use a `Result[T, E] = Ok[T] | Err[E]` sum type with `match` — callers handle both cases explicitly with no hidden control flow
- Reserve exceptions for truly exceptional conditions: programmer errors, I/O failures, invariant violations
- Never use exceptions for control flow or business logic branching

### Type system leverage

- Avoid primitive obsession — wrap distinct domain concepts (user IDs, product IDs, paths) in `@dataclass(frozen=True)` newtypes so the type checker prevents misuse at call sites
- Use modern type syntax: `list[int]`, `X | None`, `type Result[T, E] = ...` — no `typing.Optional`, `typing.List`, `typing.Union`
- Use `Protocol` for structural subtyping instead of ABC inheritance when you only need a behavioral contract

### Immutability by default

- Default to `@dataclass(frozen=True)` — only use mutable dataclasses when mutation is essential to the design
- Prefer `tuple` over `list` for fixed-size collections, `NamedTuple` for lightweight immutable records
- Return new objects from transformations instead of mutating in place — prefer `sorted()` over `.sort()`, dict comprehensions over `dict.update()`

### Composition over inheritance

- Favor protocols and delegation over deep class hierarchies — compose behaviors via injected dependencies, not inherited methods
- Inheritance is acceptable for genuine is-a relationships and framework requirements, not for code reuse

### Small, pure functions

- Functions should take explicit inputs and return outputs without relying on module-level or global state
- Pass dependencies as arguments — don't reach for globals, singletons, or module-level config objects
- Keep functions under ~30 lines; if longer, decompose by responsibility
- Classes with more than ~7 methods likely do too much — split by responsibility

---

## Anti-patterns to avoid

- **Mutable default arguments** — never use `[]`, `{}`, or `set()` as default parameter values; use `None` and create inside the function body
- **Bare or broad `except`** — never use bare `except:` or `except Exception:`; always catch the most specific exception you can handle
- **`type()` equality checks** — use `isinstance()` for type checking; `type(x) == T` breaks for subclasses
- **String concatenation in loops** — use `str.join()` or `io.StringIO` for building strings iteratively; `+=` on strings is O(n²)
- **Ignoring return values** — understand which methods mutate in place (`.sort()`, `.append()`) vs. return new values (`sorted()`, `+`); don't confuse the two
- **Premature abstraction** — don't create factories, registries, or base classes for a single use case; write the concrete function first, abstract only when a second distinct use case appears
- **`assert` for runtime validation** — `assert` is stripped by `python -O`; use `raise ValueError/TypeError` for input validation
- **Missing context managers** — always use `with` for files, locks, database connections, and any resource that needs cleanup
- **Wildcard imports** — never use `from module import *`; import specific names or use the module prefix
- **Deeply nested comprehensions** — limit comprehensions to one level of iteration; for multiple levels, extract a generator function or use explicit loops
- **Stringly-typed interfaces** — use enums, literal types, or dedicated classes instead of passing strings to control behavior (`mode="fast"` → `class Mode(Enum): FAST = auto()`)
