import numpy as np


def calculate_95CI(data, return_idx=False):
    if len(data.shape) == 1:
        data = data[np.newaxis, :]
    trial_num = data.shape[1]
    interval = int(np.ceil(trial_num * 0.25))
   
    idx = np.argsort(data, axis=1)
    med = data[np.arange(data.shape[0]), \
               idx[:, int(np.floor(trial_num / 2.0))]]
    sorted_data = np.sort(data, axis=1)
    CI_95 = sorted_data[:, (interval, trial_num-interval-1)]
    
    if return_idx:
        return med, CI_95, idx[:, int(np.floor(trial_num/2.0))]
    return med, CI_95


