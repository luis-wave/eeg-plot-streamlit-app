import streamlit as st

from pathlib import Path
import numpy as np
import pandas as pd

import mne
from scipy import signal
from mywaveanalytics.libraries import mywaveanalytics as mwa
from mywaveanalytics.libraries import (
    # database,
    # eeg_artifact_removal,
    filters,
    # eeg_computational_library,
    # ecg_statistics,
    #clinical,
    #consumer_statistics,
    references
    # protocol,
)



"""
Load the EEG data from a file
"""
@st.cache_data
def get_data_from_file_path(path, get_dict=False, picks=None):
    ext = str(Path(path).suffix)

    if ext.lower() == ".dat": eeg_type = 0
    elif ext.lower() == ".edf": eeg_type = 1

    mw_object = mwa.MyWaveAnalytics(f_path=path, eeg_type=eeg_type)
    raw_mwa = mw_object.eeg

    # raw_mwa = apply_waev_filter(raw_mwa, lowf=0.5)
    raw_mwa, _ = apply_fir_filter(
        raw_mwa, 
        fs=raw_mwa.info['sfreq'], 
        zero_phase_delay=True, 
        filter_eog=False
    )

    raw_mwa = raw_mwa.resample(128.0)
    # raw_np = raw.get_data(picks=picks)

    if get_dict is True:
        eeg_dict = get_referenced_data(raw_mwa)
        return eeg_dict
    else: 
        return raw_mwa
    
"""
Load the EEG data from mwa object
"""
# @st.cache_data
def get_data_from_mw_object(mw_object, get_dict=False, picks=None):
    raw_mwa = mw_object.eeg

    # raw_mwa = apply_waev_filter(raw_mwa, lowf=0.5)
    raw_mwa, _ = apply_fir_filter(
        raw_mwa, 
        fs=raw_mwa.info['sfreq'], 
        zero_phase_delay=True, 
        filter_eog=False
    )

    raw_mwa = raw_mwa.resample(128.0)
    # raw_np = raw.get_data(picks=picks)

    if get_dict is True:
        eeg_dict = get_referenced_data(raw_mwa)
        return eeg_dict
    else: 
        return raw_mwa

"""
All EEG data needed
"""

def get_referenced_data(raw_mwa=None):
    if raw_mwa is not None:
        raw_a1a2 = order_montage_channels(raw_mwa.copy(), "a1a2")
        raw_cz = order_montage_channels(references.centroid(raw_mwa.copy()), "cz")
        raw_bpt = bipolar_transverse(raw_mwa.copy())
        raw_tcp = references.temporal_central_parasagittal(raw_mwa.copy())
        raw_avg = order_montage_channels(references.average(raw_mwa.copy()), "avg")
        raw_ref = order_montage_channels(references.infinite_rest(raw_mwa.copy()), "ref")

        eeg_dict = dict(
            a1a2 = dict(
                raw = raw_a1a2,
                data = raw_a1a2.get_data(),
                df = raw_to_df(raw_a1a2),
                channels = raw_a1a2.info["ch_names"],
                times = raw_a1a2.times
            ),
            cz = dict(
                raw = raw_cz,
                data = raw_cz.get_data(),
                df = raw_to_df(raw_cz),
                channels = raw_cz.info["ch_names"],
                times = raw_cz.times
            ),
            bpt = dict(
                raw = raw_bpt,
                data = raw_bpt.get_data(),
                df = raw_to_df(raw_bpt),
                channels = raw_bpt.info["ch_names"],
                times = raw_bpt.times
            ),
            tcp = dict(
                raw = raw_tcp,
                data = raw_tcp.get_data(),
                df = raw_to_df(raw_tcp),
                channels = raw_tcp.info["ch_names"],
                times = raw_tcp.times
            ),
            avg = dict(
                raw = raw_avg,
                data = raw_avg.get_data(),
                df = raw_to_df(raw_avg),
                channels = raw_avg.info["ch_names"],
                times = raw_avg.times
            ),
            ref = dict(
                raw = raw_ref,
                data = raw_ref.get_data(),
                df = raw_to_df(raw_ref),
                channels = raw_ref.info["ch_names"],
                times = raw_ref.times
            ),
        )

        return eeg_dict
    else: 
        raise Exception("get_referenced_data() requires raw mne (from mwa object) to be passed in")


"""
Reorder the montage channel order
"""
def order_montage_channels(raw=None, montage=None): 
    try: 
        new_order_all = ["Fz","Cz","Pz","Fp1","Fp2","F3","F4","F7","F8", 
                        "C3","C4","T3","T4","P3","P4","T5","T6","O1","O2",
                        "A1","A2", "ECG"
                        ]
        new_order_a1a2 = ["Fz","Cz","Pz","Fp1","Fp2","F3","F4","F7","F8", 
                        "C3","C4","T3","T4","P3","P4","T5","T6","O1","O2", 
                        "A1","A2"]
        new_order_ecg = ["Fz","Cz","Pz","Fp1","Fp2","F3","F4","F7","F8", 
                        "C3","C4","T3","T4","P3","P4","T5","T6","O1","O2","ECG"]
        new_order_eeg = ["Fz","Cz","Pz","Fp1","Fp2","F3","F4","F7","F8", 
                        "C3","C4","T3","T4","P3","P4","T5","T6","O1","O2"]
        non_eeg = ["ECG", "A1", "A2"]
        remove_a1a2 = ["A1", "A2"]
        remove_ecg = ["ECG"]


        if set(["ECG", "A1", "A2"]).issubset(set(raw.info["ch_names"])):
            raw.reorder_channels(new_order_all)
        elif set(["A1", "A2"]).issubset(set(raw.info["ch_names"])):
            raw.reorder_channels(new_order_a1a2)
        elif set(["ECG"]).issubset(set(raw.info["ch_names"])):
            raw.reorder_channels(new_order_ecg)
        else:
            raw.reorder_channels(
                [channel for channel in new_order_eeg] # old condition: if channel not in remove_a1a2
            )

        
        # print(f"changing the labels of {montage} montage...")
        # Modify channel labels so the reference is displayed
        if montage == "a1a2":
            raw.drop_channels(remove_a1a2)
            old_labels = raw.info["ch_names"]
            new_labels = {
                old_label: old_label + "-A1A2" for old_label in old_labels
            }
            raw.rename_channels(new_labels)
        elif montage == "cz":
            raw.drop_channels(remove_a1a2)
            old_labels = raw.info["ch_names"]
            new_labels = {
                old_label: old_label + "-Cz" for old_label in old_labels
            }
            raw.rename_channels(new_labels)
        elif montage == "avg":
            raw.drop_channels(remove_a1a2)
            old_labels = raw.info["ch_names"]
            new_labels = {
                old_label: old_label + "-Avg" for old_label in old_labels
            }
            raw.rename_channels(new_labels)
        elif montage == "ref":
            old_labels = raw.info["ch_names"]
            new_labels = {
                old_label: old_label + "-Ref" for old_label in old_labels
            }
            raw.rename_channels(new_labels)

        return raw

    except Exception as e:
        print(f"ERROR: {e}")

    







"""
Convert an mne.io.Raw object to a pandas DataFrame

Parameters:
raw (mne.io.Raw): The raw object to convert

Returns:
pd.DataFrame: A DataFrame containing the time points and channel data
"""
def raw_to_df(raw=None):
    # Get data and times
    data = raw.get_data()  # shape (n_channels, n_times)
    times = raw.times  # shape (n_times,)
    
    # Create a DataFrame
    df = pd.DataFrame(data.T, columns=raw.ch_names)
    
    # Add the time column
    df['Time'] = times
    
    # Reorder columns to place 'Time' first
    cols = df.columns.tolist()
    cols = ['Time'] + [col for col in cols if col != 'Time']
    df = df[cols]
    
    return df


@st.cache_data
def normalize_dataframe(df, exclude_column="Time"):
    # Function to normalize between -1 and 1
    def normalize_column(column):
        return 2 * (column - column.min()) / (column.max() - column.min()) - 1

    # Normalize all columns except the exclude_column
    df_normalized = df.copy()
    columns_to_normalize = df.columns.difference([exclude_column])
    df_normalized[columns_to_normalize] = df[columns_to_normalize].apply(normalize_column)
    
    return df_normalized


"""
Low filter units: Time constant in seconds
0.08 sec is roughly equivalent to a 1.9894 Hz cuttoff frequency

highpass filter to remove signal drift e.g. : (low val=  0.50, high val = None)
"""
def apply_waev_filter(mw_object=None, 
                      time_constant=0.08, 
                      lowf=None, 
                      highf=None, 
                      default_time_constant=False,
                      ):
    
    if lowf is None and highf is None:
        print(f"Time constant used: {time_constant}")
        if default_time_constant:
            filters.eeg_filter(mw_object, 1.9894, None)
        else: 
            raise Exception("Time constant low filter in progress")
    else:
        filters.eeg_filter(mw_object, lowf, highf) 


"""
Funcion that takes a 1d or 2d array of data and applies a 1-25 Hz FIR filter
"""
def apply_fir_filter(data=[], fs=128.0, cutoff_freq=13, bandwidth=24, numtaps=200, window='blackman', zero_phase_delay=True, filter_eog=False, display_eog_data=False):
    lowcut = cutoff_freq - bandwidth / 2
    if lowcut < 0: lowcut = 0

    highcut = cutoff_freq + bandwidth / 2
    
    # design the filter
    coeffs = signal.firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs, window=window)
    
    # check the type of 'data'
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            if zero_phase_delay: filtered_data = signal.filtfilt(coeffs, 1.0, data)
            else: filtered_data = signal.lfilter(coeffs, 1.0, data)
        elif data.ndim == 2:
            if zero_phase_delay:
                filtered_data = np.apply_along_axis(lambda m: signal.filtfilt(coeffs, 1.0, m), axis=0, arr=data)
            else:
                filtered_data = np.apply_along_axis(lambda m: signal.lfilter(coeffs, 1.0, m), axis=0, arr=data)
        else:
            raise ValueError("Unsupported data shape for numpy array. Please provide 1D or 2D array.")
        
        return filtered_data, coeffs
    
    elif isinstance(data, mne.io.BaseRaw):
        # get the data and apply the filter
        original_data = data.get_data()
        
        if zero_phase_delay:
            filtered_np = np.apply_along_axis(lambda m: signal.filtfilt(coeffs, 1.0, m), axis=1, arr=original_data)
        else: 
            filtered_np = np.apply_along_axis(lambda m: signal.lfilter(coeffs, 1.0, m), axis=1, arr=original_data)
        
        # create a new MNE Raw object with the filtered data
        filtered_data = mne.io.RawArray(filtered_np, data.info)
        
        if filter_eog:
            pass

        return filtered_data, coeffs
    
    else:
        raise TypeError("Unsupported data type. Acceptable types: np.ndarray, mne.io.BaseRaw")



# Function that calculates the power spectrum and associated frequencies for a 1d or 2d array of data



# function that smooths a 1d or 2d array of power spectrum data for associated frequencies



"""
Returns helpful values for setting the y axis placement for each channel for display
"""
def get_viewer_format_values(raw=None):
    num_channels = len(raw.info["ch_names"])

    # the range sets the y axis
    range = (-0.5, num_channels-0.5)
    format_dict = dict(
        y_range = range,
        num_channels = num_channels,
        y_bottom_coordinate = 0,
        y_top_coordinate = num_channels-1,
    )

    return format_dict


def bipolar_transverse(raw) :
    # bipolar-transverse montage electrodes
    ANODES =    ['F7',
                'Fp1',
                'Fp2',
                'F7',
                'F3',
                'Fz',
                'F4',
                'T3',
                'C3',
                'Cz',
                'C4',
                'T5',
                'P3',
                'Pz',
                'P4',
                'T5',
                'O1',
                'O2']

    CATHODES =  ['Fp1',
                'Fp2',
                'F8',
                'F3',
                'Fz',
                'F4',
                'F8',
                'C3',
                'Cz',
                'C4',
                'T4',
                'P3',
                'Pz',
                'P4',
                'T6',
                'O1',
                'O2',
                'T6']
    
    # change the names of the mne raw channel names (i.e. Fp1-A1A2 -> Fp1)
    raw.rename_channels(lambda channel: channel.split('-')[0])

    # create bipolar transverse reference (BPT)
    raw_bpt = mne.set_bipolar_reference(raw, anode=ANODES, cathode=CATHODES)

    # remove irrelevant channels
    channels =  [channel 
                 for channel in raw_bpt.info['ch_names'] 
                 if "-" in channel
                 ]

    # replace with relevant channels/data
    raw_bpt.pick_channels(channels)

    return raw_bpt