import numpy as np

from chart_digitizer.config import IlluminationConfig
from chart_digitizer.stages.illumination import flatten_illumination
from chart_digitizer.synth.degrade import shadow, tint


def shaded_chart() -> np.ndarray:
    """White paper darkening toward one corner, with a thin black line across it."""
    rows, cols = np.mgrid[0:600, 0:800]
    light = 1.0 - 0.6 * (rows / 600) * (cols / 800)  # down to 40% brightness
    image = np.repeat((250 * light)[:, :, None], 3, axis=2)
    image[300:304, :] *= 0.15  # ink absorbs most of the light
    return image.astype(np.uint8)


def test_background_becomes_white_and_ink_stays_dark() -> None:
    flat = flatten_illumination(shaded_chart(), IlluminationConfig(), reference_px=600)
    background = np.delete(flat, np.s_[290:314], axis=0)
    assert background.min() >= 235
    assert flat[300:304].max() <= 60
    # Uniform: the far corner is as dark as the near one after flattening.
    assert abs(int(flat[302, 790, 0]) - int(flat[302, 10, 0])) <= 15


def test_disabled_returns_input() -> None:
    image = shaded_chart()
    assert flatten_illumination(image, IlluminationConfig(enabled=False), reference_px=600) is image


def test_hard_shadow_and_colour_cast_are_removed() -> None:
    image = np.full((600, 800, 3), 245, np.uint8)
    image[300:304, :] = 30
    image[:, 400:402] = 30
    image = tint(shadow(image, 0.5, 4.0, np.random.default_rng(1)), (0.8, 0.95, 1.1))
    flat = flatten_illumination(image, IlluminationConfig(), reference_px=600)
    paper = np.ones((600, 800), bool)
    paper[290:314] = False
    paper[:, 390:412] = False
    # Paper is near white and gray on both sides of the shadow edge, apart from a thin band along it.
    assert np.percentile(flat[paper], 2) >= 215
    assert np.abs(flat[paper].astype(int) - flat[paper].mean(axis=1, keepdims=True)).max(initial=0) <= 30
    # The lines stay dark in the shadow and out of it.
    assert flat[300:304].max(axis=2).mean() <= 90
