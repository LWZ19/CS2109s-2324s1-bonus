import numpy as np
from scipy import signal as sig

x = np.array([[0.5, 0.2, 0.1, 0.7],
              [0.1, 0.6, 0.9, 0.5],
              [0.0, 0.8, 0.2, 0.7],
              [0.2, 0.4, 0.0, 0.4]])

W = np.array([[0.1, 0.2, 0.6],
              [0.4, 0.3, 0.5],
              [0.9, 0.8, 0.7]])


def correlate2d(x, W):
    slide_window = np.lib.stride_tricks.sliding_window_view(x, W.shape)
    result = slide_window * W
    return result.sum(axis=3).sum(axis=2)


def convolve2d(x, W):
    slide_window = np.lib.stride_tricks.sliding_window_view(x, W.shape)
    flip_W = np.flip(W, axis=(0, 1))
    result = slide_window * flip_W
    return result.sum(axis=3).sum(axis=2)


expected_correlate = sig.correlate2d(x, W, mode='valid').round(3)
expected_convolve = sig.convolve2d(x, W, mode='valid').round(3)

assert (expected_correlate == correlate2d(x, W).round(3)).all()
assert (expected_convolve == convolve2d(x, W).round(3)).all()
