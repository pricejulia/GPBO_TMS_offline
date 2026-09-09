import numpy as np


class Objective:

    def __init__(self, config):
        self.config = config

    def find_mep_peak_amplitude(self, y_full, mep_window, baseline_window):

        i0, i1 = mep_window
        b0, b1 = baseline_window

        baseline_mean = float(np.nanmean(y_full[b0:b1]))
        y_win = y_full[i0:i1] - baseline_mean

        predicted_peak = float(y_win.max()) - float(y_win.min())

        return predicted_peak

    def quick_peak(self, arr):

        return self.find_mep_peak_amplitude(
            arr,
            self.config.mep_window,
            self.config.baseline_window
        )

    def func(self):

        if self.config.exp_type in ['ISI_only', 'Cint_only', 'Both']:
            return self.quick_peak

        raise NotImplementedError