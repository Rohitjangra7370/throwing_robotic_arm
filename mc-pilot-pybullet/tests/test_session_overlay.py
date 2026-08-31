import numpy as np

from session_overlay import render_overlay


def test_overlay_returns_a_side_by_side_colour_image():
    ir1 = np.zeros((480, 848), np.uint8)
    ir2 = np.zeros((480, 848), np.uint8)
    out = render_overlay(ir1, ir2)
    assert out.shape == (480, 1696, 3) and out.dtype == np.uint8


def test_overlay_draws_something_where_a_detection_is():
    ir1 = np.zeros((480, 848), np.uint8)
    ir2 = np.zeros((480, 848), np.uint8)
    blank = render_overlay(ir1, ir2)
    marked = render_overlay(ir1, ir2, detections=([(400.0, 200.0, 30.0)], []))
    assert not np.array_equal(blank, marked)
    assert marked[190:215, 385:415].any()


def test_overlay_renders_the_landing_text_when_given_one():
    ir1 = np.zeros((480, 848), np.uint8)
    out = render_overlay(ir1, ir1, landing=(0.712, 0.021, 0.018))
    assert out.any()      # text was drawn somewhere on an otherwise black frame


def test_overlay_survives_an_empty_path():
    ir1 = np.zeros((480, 848), np.uint8)
    assert render_overlay(ir1, ir1, path_px=np.zeros((0, 2))).shape == (480, 1696, 3)


def test_overlay_survives_nan_detection_coordinate():
    """NaN in u, v, or area should be skipped, not crash."""
    ir1 = np.zeros((480, 848), np.uint8)
    ir2 = np.zeros((480, 848), np.uint8)
    # Mix of valid and NaN detections
    out = render_overlay(ir1, ir2, detections=(
        [(400.0, 200.0, 30.0), (np.nan, 200.0, 30.0)],
        [(300.0, np.nan, 40.0), (500.0, 400.0, 50.0)]
    ))
    assert out.shape == (480, 1696, 3) and out.dtype == np.uint8


def test_overlay_survives_infinite_area():
    """Infinite or huge area should be capped, not crash."""
    ir1 = np.zeros((480, 848), np.uint8)
    ir2 = np.zeros((480, 848), np.uint8)
    out = render_overlay(ir1, ir2, detections=([(400.0, 200.0, np.inf)], []))
    assert out.shape == (480, 1696, 3) and out.dtype == np.uint8


def test_overlay_handles_mismatched_image_heights():
    """Mismatched heights should be padded gracefully, not crash."""
    ir1 = np.zeros((480, 848), np.uint8)
    ir2 = np.zeros((400, 848), np.uint8)  # Shorter image
    out = render_overlay(ir1, ir2)
    # Output should have the height of the taller image
    assert out.shape[0] == 480 and out.shape[1] == 1696 and out.dtype == np.uint8


def test_overlay_filters_nan_path_points():
    """NaN in path_px should be filtered out, not silently corrupted."""
    ir1 = np.zeros((480, 848), np.uint8)
    ir2 = np.zeros((480, 848), np.uint8)
    # Path with a NaN point in the middle
    path_with_nan = np.array([[10.0, 10.0], [np.nan, np.nan], [30.0, 30.0]])
    path_clean = np.array([[10.0, 10.0], [30.0, 30.0]])

    out_with_nan = render_overlay(ir1, ir2, path_px=path_with_nan)
    out_clean = render_overlay(ir1, ir2, path_px=path_clean)

    # The outputs should be identical (NaN filtered = clean path)
    assert np.array_equal(out_with_nan, out_clean)
