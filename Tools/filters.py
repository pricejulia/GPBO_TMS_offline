import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d


def filter_emg():

    def bandstop_filter(x, fs, f_low=55.0, f_high=65.0, order=1, axis=-1):
        nyq = fs / 2
        low = f_low / nyq
        high = f_high / nyq
        b, a = butter(order, [low, high], btype="bandstop")
        y = filtfilt(b, a, x, axis=axis)
        return y
    
    def bandpass_filter(x, fs, f_low=5.0, f_high=2500.0, order=1, axis=-1):
        nyq = fs / 2
        low = f_low / nyq
        high = f_high / nyq
        b, a = butter(order, [low, high], btype="band")
        y = filtfilt(b, a, x, axis=axis)
        return y

    def mov_avg(x):
        
        return uniform_filter1d(x, size=5, axis=-1, mode='nearest') * 1000.0  # mV


    def align_trials_on_last_stim(emg, isi_ms, fill_value=np.nan):
        
        emg_aligned = np.full_like(emg, fill_value, dtype=float)

        stim_idx = int(isi_ms * 6)

        emg_aligned[:-stim_idx] = emg[stim_idx:]
        
        return emg_aligned
    
    def align_tensor(emgs, isis):

        temp_emgs = emgs.reshape(-1, emgs.shape[-1])
        temp_isis = isis.flatten()

        aligned_emgs = []

        for i in range(temp_emgs.shape[0]):
            
            aligned_emgs.append(align_trials_on_last_stim(temp_emgs[i,:], temp_isis[i]))

        return np.array(aligned_emgs).reshape(emgs.shape)

    return lambda x, y : align_tensor(mov_avg(bandpass_filter(bandstop_filter(x, fs=6000, f_low=55.0, f_high=65.0, order=1), fs=6000, f_low=5.0, f_high=2500.0, order=1)),y)


