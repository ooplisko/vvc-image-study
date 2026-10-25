# Contributing

Use the locked Windows environment and keep scientific definitions explicit.
Functions are preferred for numerical operations; introduce a class only when
it owns state that several operations genuinely share. Keep the CLI thin.

```powershell
uv sync --locked
uv run --locked ruff check src tests
uv run --locked pytest -q
uv run --locked vvc-study verify
```

Tests should catch a changed numerical result, incomplete grid, wrong identity,
or broken restart behavior. They must not run full dataset encodes. Native
codec qualification is separate: `vvc-study check-codecs`.

Use small, coherent commits with an imperative subject, for example
`fix: reject incomplete JPEG matching grids` or `test: verify stopped jobs resume`.
Prefixes `feat`, `fix`, `refactor`, `test`, `docs`, and `build` are sufficient.
Describe scientific behavior changes and validation in the commit body.
Do not mix a structural refactor with altered experimental parameters.

Do not change numerical expectations merely to silence a failed test. Explain
the cause and verify corrected calculations. Keep numerical outputs in CSV/JSON
and figures in PNG. User documentation is maintained separately from generated
outputs. Editor-specific project files do not belong in the codebase.

Keep the byte-preserving rules in `.gitattributes`. Result manifests and timing
contexts hash exact file bytes, including line endings. Git must not normalize
source, configuration or result files during commit or checkout. Documentation
and workflow YAML use LF; PowerShell scripts use CRLF.
