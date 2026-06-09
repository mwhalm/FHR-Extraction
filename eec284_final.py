import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
from pathlib import Path

def get_ppg_fhr(num):
    dir_ = Path("data")
    ppg_path = dir_ / f"PPG{num}.csv"
    fhr_path = dir_ / f"FHR{num}.csv"
    ppg = pd.read_csv(ppg_path)
    fhr = pd.read_csv(fhr_path, header = None)
    fhr = fhr.iloc[:, 0].to_numpy() # convert fhr dataframe to numpy array
    return ppg, fhr
    
def bandpass(ppg, fs = 80, low = 0.2, high = 5, order = 4): 
    # bandpass filter at given frequencies
    ppg_array = ppg.to_numpy(dtype = float)
    sos = butter(N = order, Wn = [low, high], btype = "bandpass", fs = fs, output = "sos")
    ppg_filt = sosfiltfilt(sos, ppg_array, axis = 0)
    return pd.DataFrame(ppg_filt, columns = ppg.columns)

def lms(ref, mixed, order = 400, mu = 0.01):
    # do LMS noise cancelling
    ref = ref.to_numpy()
    mixed = mixed.to_numpy()
    num_samples = len(mixed)
    w = np.zeros(order)
    x = np.zeros(order)
    fetal_sig = np.zeros(num_samples)
    for i in range(num_samples):
        x[1 :] = x[: -1]
        x[0] = ref[i] # new samples at the start of the array
        y_ = np.dot(w, x) 
        error = mixed[i] - y_
        fetal_sig[i] = error
        w += (mu * error * x) / (np.dot(x, x) + 1e-9) # normalize the weight update
    return fetal_sig

def fhr_window(sig, fs = 80):
    # returns BPM with highest peak in the PSD
    f, pxx = welch(sig, fs, nperseg = len(sig), noverlap = len(sig) // 2, scaling = "density")
    # only interested in typical FHR range
    range_ = (f >= 1.83) & (f <= 4.5)
    f_interest = f[range_]
    return f_interest[np.argmax(pxx[range_])] * 60
    
def estimate_fhr(sig, fs = 80):
    window = 60 * fs # 60 second window
    step = 30 * fs # 30 second step size
    fhr = []
    time = []
    # find peak frequency in the PSD in a 60s window with 30s overlap
    for i in range(0, len(sig) - window, step):
        fhr.append(fhr_window(sig[i : i + window], fs = fs))
        time.append((i + window // 2) // fs)
    return np.array(time), np.array(fhr)

def weighted_median(vals, weights):
    vals = np.asarray(vals, dtype = float)
    weights = np.asarray(weights, dtype = float)
    order = np.argsort(vals)
    vals = vals[order]
    weights = weights[order]
    cumsum = np.cumsum(weights)
    threshold = 0.5 * np.sum(weights)
    return vals[np.searchsorted(cumsum, threshold)]
    
def merge_fhr(fhr_values, weights = np.array([1, 3, 2, 2])):
    fhr_values = np.asarray(fhr_values, dtype = float)
    weights = np.asarray(weights, dtype = float)
    med = weighted_median(fhr_values, weights)
    deviations = np.abs(fhr_values - med)
    MAD = np.median(deviations)
    keep = deviations <= 2 * MAD
    return np.average(fhr_values[keep], weights = weights[keep])

def merge_all(fhr_channels): # merge FHR channels with outlier detection
    merged = []
    # each row contains the estimated fhr of the four channels at a specific time
    for row in fhr_channels: 
        merged.append(merge_fhr(row))
    return np.array(merged)
    
def merge_wavelengths_weighted(wavelength_1, wavelength_2, window):
    wavelength_1 = np.asarray(wavelength_1, dtype = float)
    wavelength_2 = np.asarray(wavelength_2, dtype = float)
    final = []
    # merge wavelengths based on a previous sample window
    # more weight in the average the lower the variance is (inversely proportional)
    for i in range(len(wavelength_1)):
        start = max(0, i - window + 1) # ensures that indices at the start are valid
        var1 = np.var(wavelength_1[start : i + 1])
        var2 = np.var(wavelength_2[start : i + 1])
        weight_1 = 1 / (var1 + 1e-12)
        weight_2 = 1 / (var2 + 1e-12)
        vals = np.array([wavelength_1[i], wavelength_2[i]])
        weights = np.array([weight_1, weight_2])
        final.append(np.average(vals, weights = weights))
    return np.array(final)

def compute_rmse(estimated_time, estimated_fhr, fhr_ref):
    time_index = 0
    sum_ = 0
    for i in range(len(fhr_ref)): # 1 second per data point
        # only do computation when there is a sample at a specific time
        if time_index < len(estimated_time) and i == int(estimated_time[time_index]):
            sum_ += (estimated_fhr[time_index] - fhr_ref[i]) ** 2
            time_index += 1
    return np.sqrt(sum_ / len(estimated_fhr))

def compute_mae(estimated_time, estimated_fhr, fhr_ref):
    time_index = 0
    sum_ = 0
    for i in range(len(fhr_ref)): # 1 second per data point
        # only do computation when there is a sample at a specific time
        if time_index < len(estimated_time) and i == int(estimated_time[time_index]):
            sum_ += abs(estimated_fhr[time_index] - fhr_ref[i])
            time_index +=1
    return sum_ / len(estimated_fhr)
    
def plot_estimates(ppg, fhr_ref, num, order):
    fhr_detectors = {2: [], 3: [], 4: [], 5: []}
    plt.figure(figsize = (10, 4))
    print(f"\n---------- FHR {num} Metrics ----------\n")
    for j in range(1, 3):
        maternal = ppg[f"ch1voltsWL{j}"]
        for i in range(2, 6):
            mixed = ppg[f"ch{i}voltsWL{j}"]
            fetal_sig = lms(maternal, mixed, order) # LMS noise cancelling 
            time, fhr_temp = estimate_fhr(fetal_sig) # estimate FHR using 60s window with 30s overlap
            fhr_detectors[i].append(fhr_temp)

    combined_detectors = []
    for i in range(2, 6):
        # combine wavelengths with weights determined by the variance of the previous 5 samples
        combined = merge_wavelengths_weighted(fhr_detectors[i][0], fhr_detectors[i][1], window = 5) 
        combined_detectors.append(combined)
    combined_detectors = np.array(combined_detectors)
    
    for i in range(2, 6):
        row = i - 2
        curr_channel = combined_detectors[row, :]
        rmse = compute_rmse(time, curr_channel, fhr_ref) # compute RMSE per detector
        mae = compute_mae(time, curr_channel, fhr_ref) # compute MAE per detector
        print(f"RMSE D{i}: {rmse:.2f}\tMAE D{i}: {mae:.2f}")
        plt.plot(time / 60, curr_channel, label = f"D{i}", linestyle = "--", linewidth = 2, alpha = 0.5)  
    final_fhr = merge_all(np.column_stack(combined_detectors)) # merge all detectors
    rmse = compute_rmse(time, final_fhr, fhr_ref) # compute RMSE for final combined FHR
    mae = compute_mae(time, final_fhr, fhr_ref) # compute MAE for final combined FHR
    print(f"RMSE Merged: {rmse:.2f}\tMAE Merged: {mae:.2f}")
    plt.plot(time / 60, final_fhr, label = f"Merged FHR", linewidth = 3)
    plt.plot(np.arange(0, len(fhr_ref)) / 60, fhr_ref, label = "Reference FHR", linewidth = 3)
    plt.xlabel("Time (min)")
    plt.ylabel("FHR (BPM)")
    plt.title(f"PPG{num}: Estimated vs Reference\n")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"FHR{num}.png", dpi = 300, bbox_inches = "tight")
    plt.show()


# set sampling frequency
fs = 80

# put ppg signals into pandas dataframe
# put fhr signals in a numpy array
ppg1, fhr1 = get_ppg_fhr(1) 
ppg2, fhr2 = get_ppg_fhr(2)
ppg3, fhr3 = get_ppg_fhr(3)

# bandpass filter all channels at f_low = 0.2 Hz and f_high = 5 Hz
ppg1 = bandpass(ppg1)
ppg2 = bandpass(ppg2)
ppg3 = bandpass(ppg3)

# plot the estimated fhr for each ppg
order = 400
plot_estimates(ppg1, fhr1, 1, order)
plot_estimates(ppg2, fhr2, 2, order)
plot_estimates(ppg3, fhr3, 3, order)