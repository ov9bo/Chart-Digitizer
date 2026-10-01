# Contributing

Thanks for helping. Bug reports that include a chart that fails are the most useful thing you can send.

## Reporting a chart that doesn't digitize

Open an issue with:

- the image, if you can share it (crop or blur anything private), or a description of the chart;
- the exact command or web-app settings, and the axis ranges;
- the error message, or `overlay.png` if the result is wrong;
- if possible, the `debug/` images from a run with `--debug`, which show where it went wrong.

Please don't post datasheets or figures you aren't allowed to share.

## Ground rules

The project has one hard constraint: **classical computer vision only.** No OCR, no neural
networks and no learned models of any kind, and no network calls at runtime. Pull requests that add
any of these won't be merged, however good the results.

Beyond that:

- Small, pure functions with type hints; no global state.
- Fail loudly: raise a stage error with a message that says what went wrong and a hint for what to
  try. Never return a silently wrong answer.
- New thresholds go in the config models in `src/chart_digitizer/config.py`, as fractions of the
  plot size rather than pixel constants in the code.
- Add a test. `digitize-synth` renders charts with exact ground truth, which makes accuracy
  regressions easy to catch.

## Setting up

```bash
uv sync
uv run pytest -m "not slow"   # quick
uv run pytest                 # full suite, about two minutes
```

`docs/SPEC.md` describes the pipeline and the reasoning behind it.
