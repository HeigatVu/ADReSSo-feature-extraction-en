import math
import numpy as np
import parselmouth
from parselmouth.praat import call
import pandas as pd


# Feature Utils
## Prosody and Fluency
def get_intensity_attributes(audio_file:parselmouth.Sound, 
                                time_step:float=0.0, 
                                pitch_floor:float=75.0,
                                interpolation:str="Parabolic",
                                return_values:bool=False,
                                replacement_for_nan:float=0.0,
                                ) -> tuple[dict, list]:
    """ 
    Function to get intensity attributes such as minimum intensity, maximum intensity, mean
    intensity, and standard deviation of intensity.
    """
    actual_start = call(audio_file, "Get start time")
    actual_end = call(audio_file, "Get end time")
    duration = actual_end - actual_start
    
    intensity = call(audio_file, "To Intensity", pitch_floor, time_step, "yes")
    attributes = dict()
    
    query_start, query_end = 0.0, 0.0 # Passing 0.0, 0.0 tells Praat to use entire the audio
    
    attributes["min_intensity"] = call(intensity, "Get minimum", query_start, query_end, interpolation)

    abs_min_time = call(intensity, "Get time of minimum", query_start, query_end, interpolation)
    if duration > 0:
        attributes["relative_min_intensity_time"] = (abs_min_time - actual_start) / duration
    else:
        attributes["relative_min_intensity_time"] = 0.0

    attributes["mean_intensity"] = call(intensity, "Get mean", query_start, query_end)
    attributes["stddev_intensity"] = call(intensity, "Get standard deviation", query_start, query_end)
    attributes["q1_intensity"] = call(intensity, "Get quantile", query_start, query_end, 0.25)
    attributes["median_intensity"] = call(intensity, "Get quantile", query_start, query_end, 0.50)
    attributes["q3_intensity"] = call(intensity, "Get quantile", query_start, query_end, 0.75)
    
    intensity_values = None

    if return_values:
        intensity_values = []
        num_frames = call(intensity, "Get number of frames")
        for frame_no in range(1, num_frames+1): 
            value = call(intensity, "Get value in frame", frame_no)
            if math.isnan(value):
                intensity_values.append(replacement_for_nan)
            else:
                intensity_values.append(value)
    
    return attributes, intensity_values


def get_pitch_attributes(audio_file:parselmouth.Sound,
                            pitch_type:str="preferred",
                            time_step:float=0.0,
                            pitch_floor:float=75.0,
                            pitch_ceiling:float=600.0,
                            unit:str="Hertz",
                            interpolation:str="Parabolic",
                            return_values:bool=False,
                            replacement_for_nan:float=0.0,
                        ) -> tuple[dict, list]:
    """
    Function to get pitch attributes such as minimum pitch, maximum pitch, mean pitch, and
    standard deviation of pitch.
    """
    actual_start = call(audio_file, "Get start time")
    actual_end = call(audio_file, "Get end time")
    duration = actual_end - actual_start

    # Create pitch object
    if pitch_type == 'preferred':
        pitch = call(audio_file, 'To Pitch', time_step, pitch_floor, pitch_ceiling)
    elif pitch_type == 'cc':
        pitch = call(audio_file, 'To Pitch (cc)', time_step, pitch_floor, pitch_ceiling)
    else:
        raise ValueError('Argument for @pitch_type not recognized!')

    attributes = dict()
    query_start, query_end = 0.0, 0.0
    
    attributes["voice_fraction"] = call(pitch, "Count voiced frames") / len(pitch)
    attributes["min_pitch"] = call(pitch, "Get minimum", query_start, query_end, unit, interpolation)
    
    abs_min_time = call(pitch, "Get time of minimum", query_start, query_end, unit, interpolation)
    if duration > 0:
        attributes["relative_min_pitch_time"] = (abs_min_time - actual_start) / duration
    else:
        attributes["relative_min_pitch_time"] = 0.0

    attributes["max_pitch"] = call(pitch, "Get maximum", query_start, query_end, unit, interpolation)
    
    abs_max_time = call(pitch, "Get time of maximum", query_start, query_end, unit, interpolation)
    if duration > 0:
        attributes["relative_max_pitch_time"] = (abs_max_time - actual_start) / duration
    else:
        attributes["relative_max_pitch_time"] = 0.0

    attributes["mean_pitch"] = call(pitch, "Get mean", query_start, query_end, unit)
    attributes["stddev_pitch"] = call(pitch, "Get standard deviation", query_start, query_end, unit)
    attributes["q1_pitch"] = call(pitch, "Get quantile", query_start, query_end, 0.25, unit)
    attributes["median_pitch"] = call(pitch, "Get quantile", query_start, query_end, 0.50, unit)
    attributes["q3_pitch"] = call(pitch, "Get quantile", query_start, query_end, 0.75, unit)
    attributes['mean_absolute_pitch_slope'] = call(pitch, 'Get mean absolute slope', unit)
    attributes['pitch_slope_without_octave_jumps'] = call(pitch, 'Get slope without octave jumps')

    pitch_values = None
    if return_values:
        pitch_values = []
        num_frames = call(pitch, "Get number of frames")
        for frame_no in range(1, num_frames+1):
            value = call(pitch, "Get value in frame", frame_no, unit)
            if math.isnan(value):
                pitch_values.append(replacement_for_nan)
            else:
                pitch_values.append(value)
    
    return attributes, pitch_values


def get_speaking_rate(audio_path:str,
                        transcript:str="",
                        ) -> float:
    """
    Function to get speaking rate, approximated as number of words divided by total duration.
    """
    audio_file = parselmouth.Sound(audio_path)
    duration = call(audio_file, 'Get end time')
    word_count = len(str(transcript).split())
    return word_count / duration if duration > 0 else 0

## Voice Quality and Phonation
# def get_harmonics_to_noise_ratio_attributes(
#                                             audio_file:parselmouth.Sound,
#                                             harmonic_type:str="preferred",
#                                             time_step:float=0.01,
#                                             min_time:float=0.0,
#                                             max_time:float=0.0,
#                                             minimum_pitch:float=0.75,
#                                             silence_threshold:float=0.0,
#                                             num_periods_per_window:float=1.0,
#                                             interpolation:str="Parabolic",
#                                             return_values:bool=False,
#                                             replacement_for_nan:float=0.0,
#                                             ):
#     """
#     Function to get Harmonics-to-Noise Ratio (HNR) attributes such as minimum HNR, maximum HNR,
#     mean HNR, and standard deviation of HNR. HNR is defined as a measure that quantifies the amount
#     of additive noise in a voice signal.
#     """
#     pass

# def get_glottal_to_noise_ratio_attributes(audio_file:parselmouth.Sound,
#                                             horizontal_minimum:float=0.0,
#                                             horizontal_maximum:float=0.0,
#                                             vertical_minimum:float=0.0,
#                                             vertical_maximum:float=0.0,
#                                             minimum_frequency:float=500.0,
#                                             maximum_frequency:float=4500.0,
#                                             bandwidth:float=1000.0,
#                                             step:int=50,
#                                             ):
#     """
#     Function to get Glottal-to-Noise Ratio (GNE) attributes such as minimum GNE, maximum GNE,
#     mean GNE, standard deviation of GNE, and sum of GNE. GNE is a measure that indicates whether a
#     given voice signal originates from vibrations of the vocal folds or from turbulent noise
#     generated in the vocal tract and is thus related to (but not a direct measure of) breathiness.
#     """
#     pass

def get_local_jitter(audio_file:parselmouth.Sound,
                        pitch_floor:float=75.0,
                        pitch_ceiling:float=600.0,
                        period_floor:float=0.0001,
                        period_ceiling:float=0.02,
                        max_period_factor:float=1.3,
                        ) -> float:
    """    
    Function to calculate (local) jitter from a periodic PointProcess.
    """
    
    query_start = 0.0
    query_end = 0.0
    point_process = call(audio_file, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    local_jitter = call(point_process, "Get jitter (local)", query_start, query_end, period_floor, period_ceiling, max_period_factor)
        
    return local_jitter

def get_local_shimmer(audio_file:parselmouth.Sound,
                        pitch_floor:float=75.0,
                        pitch_ceiling:float=600.0,
                        period_floor:float=0.0001,
                        period_ceiling:float=0.02,
                        max_period_factor:float=1.3,
                        max_amplitude_factor:float=1.6,
                        ) -> float:
    """
    Function to calculate (local) shimmer from a periodic PointProcess.
    """
    
    query_start = 0.0
    query_end = 0.0
    point_process = call(audio_file, 'To PointProcess (periodic, cc)', pitch_floor, pitch_ceiling)
    local_shimmer = call([audio_file, point_process], 'Get shimmer (local)', 
                            query_start, query_end, period_floor, period_ceiling, 
                            max_period_factor, max_amplitude_factor)
    
    return local_shimmer

# Spectral and Articulatory
def get_spectrum_attributes(audio_file:parselmouth.Sound,
                                band_floor:float=200.0,
                                band_ceiling:float=1000.0,
                                low_band_floor:float=0.0,
                                low_band_ceiling:float=500.0,
                                high_band_floor:float=500.0,
                                high_band_ceiling:float=4000.0,
                                power:float=2.0,
                                moment:float=3.0,
                                return_values:bool=False,
                                replacement_for_nan:float=0.0,
                            ):
    """
    Function to get spectrum-based attributes such as center of gravity, skewness, kurtosis, etc.
    """
    spectrum = call(audio_file, "To Spectrum", "yes")
    attributes = dict()

    attributes["band_energy"] = call(spectrum, "Get band energy", band_floor, band_ceiling)
    attributes["band_density"] = call(spectrum, "Get band density", band_floor, band_ceiling)
    attributes["band_energy_difference"] = call(spectrum, "Get band energy difference", 
                                                low_band_floor, low_band_ceiling,
                                                high_band_floor, high_band_ceiling)
    

def get_formant_attributes(audio_file:parselmouth.Sound,
                            time_step:float=0.0,
                            pitch_floor:float=75.0,
                            pitch_ceiling:float=600.0,
                            max_num_formants:float=5.0,
                            max_formant:float=5500.0,
                            window_length:float=0.025,
                            pre_emphasis_frequency:float=50.0,
                            unit:str="Hertz",
                            interpolation:str="Linear",
                            replacement_for_nan:float=0.0,
                            ):
    """
    Function to get formant-related attributes such as mean and median formants.
    NOTE: All frequency units are 'Hertz' in this function.
    """
    pass


## Cepstral (Timbral)
def get_mfcc(audio_file:parselmouth.Sound,
            window_length:float=0.015,
            tkme_step:float=0.005,
            first_filter_frequency:float=100.0,
            distance_between_filters:float=100.0,
            maximum_frequency:float=0.0,
            ):
    """
    Function to calculate the MFCC (Mel Frequency Cepstral Coefficients).
    """
    pass

# def get_lfcc(audio_file:parselmouth.Sound,
#                 lpc_method:str="autocorrelation",
#                 prediction:int=16,
#                 window_length:float=0.025,
#                 time_step:float=0.005,
#                 pre_emphasis_frequency:float=50.0,
#                 num_coefficients:int=12,
#             ):
#     """
#     Function calculate LFCC (Linear Frequency Cepstral Coefficients).   
#     """
#     pass

# def get_delta(matrix:np.ndarray,
#                 step_size:int=2,
#                 ):
#     """
#     Function to get a delta matrix on a given matrix, adapted from:
#     http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
#     If get the delta of a MFCC matrix -> get the velocity of MFCC.
#     If get the delta on this resulting velocity -> get the acceleration of MFCCs.
#     """
#     pass