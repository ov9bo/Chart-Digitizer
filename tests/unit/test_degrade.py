import numpy as np

from chart_digitizer.synth.degrade import DegradeSpec, degrade, shadow, tint

WHITE = np.full((200, 300, 3), 200, np.uint8)


def test_shadow_darkens_one_side_of_an_edge() -> None:
    shaded = shadow(WHITE, 0.5, 6.0, np.random.default_rng(1))[:, :, 0].astype(int)
    assert shaded.min() == 100 and shaded.max() == 200
    # Both the lit and the shadowed side cover a good part of the image.
    assert 0.1 < (shaded == 100).mean() < 0.9
    assert 0.1 < (shaded == 200).mean() < 0.9


def test_tint_scales_channels() -> None:
    assert tuple(tint(WHITE, (0.5, 1.0, 1.5))[0, 0]) == (100, 200, 255)


def test_lighting_options_default_off_and_keep_other_degradations_unchanged() -> None:
    image = np.full((120, 160, 3), 230, np.uint8)
    corners = np.array([[20, 100], [140, 100], [140, 20], [20, 20]], dtype=float)
    plain, _ = degrade(image, corners, DegradeSpec(), np.random.default_rng(0))
    again, _ = degrade(image, corners, DegradeSpec(tint=(1.0, 1.0, 1.0)), np.random.default_rng(0))
    assert np.array_equal(plain, again)
    shaded, shaded_corners = degrade(image, corners, DegradeSpec(shadow=0.5), np.random.default_rng(0))
    assert shaded.mean() < plain.mean() - 10
