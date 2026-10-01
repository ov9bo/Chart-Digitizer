# chart-digitizer: specification

This is the original project brief, followed by the decisions agreed in Phase 0.

## Goal

A local, fully offline Python tool that takes a photo or screenshot of a 2D line chart and outputs
the digitized curve as data points plus a clean re-plot. It uses no AI or ML models of any kind: no
LLM calls, no neural OCR, no learned models. Everything is classical computer vision (OpenCV + numpy).

## Working agreement

Work in phases. At the end of each phase, stop, summarize what was built and any design decisions,
and wait for approval before continuing.

## Reference input

`Samples/20260923_174008.jpg` (a private photo, not distributed with the repository; its test skips
when it is absent): a valve characteristic curve, Kv/Kvs [%] vs Stroke [%], both axes
0–100, gridlines every 10. It's a phone photo of a monitor: perspective skew, uneven lighting and
vignetting, moiré, dust specks, and a mouse cursor on the plot. It contains:

- the main solid black curve (the target);
- straight horizontal and vertical reference lines labelled "min" and "norm" that intersect the curve;
- a dashed red/brown near-horizontal line near y≈2 (Kv0), which must NOT be picked up;
- text labels inside the plot area.

The curve has a slope kink around x≈40. Ground truth lives in `tests/fixtures/samples_truth.yaml`.

## Stack

Python 3.11+, managed with `uv`: opencv-python, numpy, scipy, matplotlib, typer, pydantic, pytest.
No Tesseract or any OCR; the user supplies axis ranges.

## Pipeline

1. **Load & normalize**: grayscale, illumination correction (divide by large blur or morphological
   background).
2. **Plot-area detection**: find the four plot corners from the outermost gridline/axis frame.
   Fallbacks: `--corners x1,y1,...,x4,y4` or `--interactive`. Corners map to (xmin,ymin),
   (xmax,ymin), (xmax,ymax), (xmin,ymax).
3. **Perspective correction**: homography warp to an axis-aligned rectangle (default 2000×2000).
4. **Calibration**: pixel→data mapping from `--x-range`/`--y-range`; `--x-log`/`--y-log`.
5. **Clutter removal**: gridlines, straight reference lines (keeping curve pixels at crossings),
   dashed lines, specks, cursor, and text.
6. **Curve selection**: dominant dark curve by default; `--curve-color` for colored curves. One curve
   in v1, structured for multi-curve later.
7. **Tracing**: column-wise with continuity (dynamic programming, max-jump constraint), PCHIP gap
   filling, filled regions flagged.
8. **Resampling**: `--points N` (default 50), `--x-values`, or `--step`.
9. **Outputs** in `out/<image-stem>/`: `points.csv`, `points.json`, `overlay.png`, `replot.png`,
   and `debug/` with an image from every stage when `--debug` is set.

## Testing

- Synthetic chart generator (matplotlib) with exact ground truth, plus degradations (perspective,
  blur, JPEG, noise, vignetting, moiré).
- RMSE < 1% of y-range on clean synthetic charts, < 2% on degraded ones, with correct corners.
  Separate tests for auto corner detection.
- Regression test on the reference sample using the hand-read points (±1.5).

## Phases

- **Phase 0**: module layout, CLI, config schema.
- **Phase 1**: scaffold, synthetic generator, stages 3–4 + 7–9 with manual corners on clean
  synthetic images.
- **Phase 2**: clutter removal (5) and curve selection (6); degraded synthetic + sample; show overlay.
- **Phase 3**: auto plot-area detection (2), illumination correction (1), interactive fallback.
- **Phase 4**: batch mode, log axes, README with usage and known limitations.

## Standards

Type hints everywhere, small pure functions per stage, no global state. Tunable thresholds live in
the config with defaults. When a stage fails, exit with a clear message suggesting the fallback flag.
Never fail silently.

## Decisions agreed in Phase 0

- **Ground truth correction.** The brief's hand-read (90, 69.2) doesn't match the image: the curve
  crosses the "norm" gridline at y≈74. We use ≈73.8 (to be re-measured in Phase 2). The brief's
  (100, 100) is dropped because the curve leaves the top of the plot at x≈98.
- **Regression sample.** Only `20260923_174008.jpg`. The other photos in `Samples/` are out of
  scope for v1 (dashed gridlines, tick-only axes, markers and callouts, colored reference lines,
  frame wider than the data).
- **Merge order.** Built-in defaults, then the YAML file, then CLI flags. The `sampling` and
  `corners` sections are replaced as a whole when the CLI sets any of their keys.
- **Lengths as fractions.** Tunable lengths are fractions of the rectified plot size.
- **Padding.** The rectified image keeps a margin (`rectify.pad_frac`) outside the axes box so
  curve pixels on the axes survive.
- **No extrapolation.** Points are resampled only over the traced x-extent. Requested x values
  outside it get an empty y and `interpolated_flag=out_of_range`, plus a warning.
- **Function of x only.** v1 supports only curves that are a function of x.
- **Exit codes.** 0 on success, 1 when a stage fails (the message names the fallback flag), 2 on a
  usage or config error.
- **No git for now.**

## Decisions agreed in Phase 4

- **Per-image YAML.** A file with the image's stem next to it (`chart1.png` → `chart1.yaml`, then
  `.yml`) is merged between `--config` and the CLI flags, in both single-image and batch mode. The
  tool prints a line whenever it uses one.
- **Batch input.** A folder is processed flat (no subfolders), in name order, for the suffixes
  `.png .jpg .jpeg .bmp .tif .tiff .webp`. Images sharing a stem get output folders `stem_ext`.
- **Validate first, then keep going.** All per-image settings are validated before any image is
  processed. Any invalid settings stop the whole batch, with every problem listed (exit 2). After
  that, a failing image is reported and the batch continues. Stage failures are `failed` and
  unexpected exceptions are `error`, and neither is hidden.
- **Batch exit code.** 0 if every image succeeded, 1 if any failed, 2 for an empty folder or
  invalid settings.
- **summary.csv** in the output root: `image, status, output, points_kept, points_total, corners,
  filled_fraction, warnings, error`.
- **Log axes.** Calibration is linear in log10. `--points` is geometric on a log x axis. `--step`
  stays a linear step in data units (documented). `--x-values` must be positive on a log x axis.
  The re-plot uses log scales.
- **README** documents usage, configuration, batch mode, accuracy and known limitations.
  `examples/` holds three synthetic charts with matching YAML files, and `examples/sample.yaml` is a
  commented config; unit tests keep them valid.
