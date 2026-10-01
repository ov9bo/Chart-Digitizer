# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] - 2026-10-01

First public release.

### Added

- `digitize` command: automatic plot-frame detection, perspective correction, illumination
  flattening, clutter removal, curve tracing and resampling, with CSV, JSON, overlay and re-plot
  outputs.
- Linear and logarithmic axes, coloured-curve selection, and manual or interactive corners.
- Batch mode over a folder, with per-image YAML settings and `summary.csv`.
- `digitize-ui`: a local web app with a corner editor, linked chart and table, and debug stages.
- `digitize-synth`: synthetic charts with exact ground truth, used by the test suite.

[0.1.0]: https://github.com/ov9bo/Chart-Digitizer/releases/tag/v0.1.0
