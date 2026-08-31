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
