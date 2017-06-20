import os
import logging
from summary.utils import accumulate
from summary.utils import clean_outliner
from summary.utils import plot_correlation
from summary.nolinear_summary import get_one_file_complex_correlation
import numpy as np


def test_if_one_file_fit_summary(text, title):
    mini_length = 200
    if not os.path.isfile(text) and len(text) < mini_length:
        return False, -1
    else:
        complex_correlation = get_one_file_complex_correlation(text, title)
        return have_main_point(complex_correlation, plot=False)


def have_main_point(complex_correlation, plot=True):
    if len(complex_correlation) < 10:
        return False, -1
    else:
        complex_correlation = clean_outliner(complex_correlation)
        k = (1.0 - 0) / (len(complex_correlation) - 0)
        ys = k * np.arange(0, len(complex_correlation))
        acc = accumulate(complex_correlation)
        if plot: plot_correlation(acc)
        l2_loss = l_2_loss(acc, ys)
        threshold = 5.0e-4
        logging.info(l2_loss)
        if l2_loss < threshold:
            return False, l2_loss
        else:
            return True, l2_loss


def l_2_loss(y_hats, ys):
    def f(x):
        max_length = 1000
        if x > max_length: return max_length
        else: return x
    return np.sum(np.square(y_hats - ys)) * 1 / f(len(y_hats))

