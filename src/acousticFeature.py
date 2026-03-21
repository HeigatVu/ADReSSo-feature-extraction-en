import math
import numpy as np
import parselmouth
from parselmouth.praat import call
import opensmile
import librosa


# Feature PRAAT
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
    
    # Converting audio to intensity
    intensity = call(audio_file, "To Intensity", pitch_floor, time_step, "yes")
    attributes = dict()
    
    query_start, query_end = 0.0, 0.0 # Passing 0.0, 0.0 tells Praat to use entire the audio
    
    # Adding attribute to dictionary
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
    
    # Adding attribute to dictionary
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
                            ) -> tuple[dict, list]:
    """
    Function to get spectrum-based attributes such as center of gravity, skewness, kurtosis, etc.
    """
    # Converting audio to spectrum
    spectrum = call(audio_file, "To Spectrum", "yes")
    attributes = dict()
    # Adding attribute to dictionary
    attributes["band_energy"] = call(spectrum, "Get band energy", band_floor, band_ceiling)
    attributes["band_density"] = call(spectrum, "Get band density", band_floor, band_ceiling)
    attributes["band_energy_difference"] = call(spectrum, "Get band energy difference", 
                                                low_band_floor, low_band_ceiling,
                                                high_band_floor, high_band_ceiling)
    attributes["band_density_difference"] = call(spectrum, "Get band density difference", 
                                                low_band_floor, low_band_ceiling,
                                                high_band_floor, high_band_ceiling)
    attributes["center_of_gravity_spectrum"] = call(spectrum, "Get centre of gravity", power)
    attributes["stddev_spectrum"] = call(spectrum, "Get standard deviation", power)
    attributes["skewness_spectrum"] = call(spectrum, "Get skewness", power)
    attributes["kurtosis_spectrum"] = call(spectrum, "Get kurtosis", power)
    attributes["central_moment_spectrum"] = call(spectrum, "Get central moment", moment, power)

    spectrum_values = None
    if return_values:
        spectrum_values = []
        for bin_no in range(len(spectrum)):
            spectrum_value = call(spectrum, "Get real value in bin", bin_no)
            if math.isnan(spectrum_value):
                spectrum_values.append(replacement_for_nan)
            else:
                spectrum_values.append(spectrum_value)
    return attributes, spectrum_values

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
    # Converting audio to point process
    point_process = call(audio_file, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
    # Converting audio to formant
    formant = call(audio_file, "To Formant (burg)", time_step, max_num_formants, 
                                                            max_formant, window_length, pre_emphasis_frequency)
    
    num_points = call(point_process, "Get number of points")
    if num_points == 0:
        return dict(), None
    
    f1_list, f2_list, f3_list, f4_list = [], [], [], []

    # Meansure formants
    for point in range(1, num_points + 1):
        time = call(point_process, "Get time from index", point)
        f1 = call(formant, "Get value at time", 1, time, unit, interpolation)
        f2 = call(formant, "Get value at time", 2, time, unit, interpolation)
        f3 = call(formant, "Get value at time", 3, time, unit, interpolation)
        f4 = call(formant, "Get value at time", 4, time, unit, interpolation)
        
        if math.isnan(f1):
            f1 = replacement_for_nan
        if math.isnan(f2):
            f2 = replacement_for_nan
        if math.isnan(f3):
            f3 = replacement_for_nan
        if math.isnan(f4):
            f4 = replacement_for_nan
        
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)

        attributes = dict()
        # Calculate mean, median formants across pulse
        attributes["f1_mean"] = np.mean(f1_list)
        attributes["f2_mean"] = np.mean(f2_list)
        attributes["f3_mean"] = np.mean(f3_list)
        attributes["f4_mean"] = np.mean(f4_list)
        attributes["f1_median"] = np.median(f1_list)
        attributes["f2_median"] = np.median(f2_list)
        attributes["f3_median"] = np.median(f3_list)
        attributes["f4_median"] = np.median(f4_list)

        attributes["formant_dispersion"] = (attributes["f4_median"] - attributes["f1_median"]) / 3
        attributes["average_formant"] = (attributes["f1_median"] + attributes["f2_median"] + 
                                        attributes["f3_median"] + attributes["f4_median"]) / 4
        attributes["mff"] = (attributes["f1_median"] * attributes["f2_median"] * 
                            attributes["f3_median"] * attributes["f4_median"]) ** 0.25
        # attributes["fitch_vtl"] = ((1 * (35000 / (4 * attributes["f1_median"] + 1e-5))) + 
        #                             (3 * (35000 / (4 * attributes["f2_median"] + 1e-5))) + 
        #                             (5 * (35000 / (4 * attributes["f3_median"] + 1e-5))) + 
        #                             (7 * (35000 / (4 * attributes["f4_median"] + 1e-5)))) / 4
        # xy_sum = ((0.5 * attributes["f1_median"]) + (1.5 * attributes["f2_median"]) + 
        #             (2.5 * attributes["f3_median"]) + (3.5 * attributes["f4_median"]))
        # x_squared_sum = ((0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2))
        # attributes["delta_f"] = xy_sum / (x_squared_sum + 1e-5)
        # attributes["vtl_delta_f"] = 35000 / (2 * attributes["delta_f"] + 1e-5)
        
        if attributes["f1_median"] == 0 or attributes["f2_median"] == 0 or attributes["f3_median"] == 0 or attributes["f4_median"] == 0:
            attributes["fitch_vtl"] = 0.0
            attributes["delta_f"] = 0.0
            attributes["vtl_delta_f"] = 0.0
        else:
            attributes["fitch_vtl"] = (1 * (35000 / (4 * attributes["f1_median"])) + 
                                       3 * (35000 / (4 * attributes["f2_median"])) + 
                                       5 * (35000 / (4 * attributes["f3_median"])) + 
                                       7 * (35000 / (4 * attributes["f4_median"]))) / 4
            
            xy_sum = ((0.5 * attributes["f1_median"]) + (1.5 * attributes["f2_median"]) + 
                        (2.5 * attributes["f3_median"]) + (3.5 * attributes["f4_median"]))
            x_squared_sum = ((0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2))
            
            attributes["delta_f"] = xy_sum / x_squared_sum
            attributes["vtl_delta_f"] = 35000 / (2 * attributes["delta_f"])


    return attributes, None

## Cepstral (Timbral)
def get_mfcc(audio_file:parselmouth.Sound,
            num_coefficients:int=12,
            window_length:float=0.015,
            time_step:float=0.005,
            first_filter_frequency:float=100.0,
            distance_between_filters:float=100.0,
            maximum_frequency:float=0.0,
            ):
    """
    Function to calculate the MFCC (Mel Frequency Cepstral Coefficients).
    """
    mfcc = call(audio_file, "To MFCC", num_coefficients, window_length, time_step,
                first_filter_frequency, distance_between_filters, maximum_frequency)

    num_frames = call(mfcc, "Get number of frames")
    mfcc_matrix = np.zeros((num_frames, num_coefficients))
    for frame in range(1, num_frames + 1):
        for coefficient in range(1, num_coefficients + 1):
            mfcc_matrix[frame - 1, coefficient - 1] = call(mfcc, "Get value at time", frame, coefficient)

    return mfcc_matrix

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



# Feature openSMILE
def get_opensmile_features(signal:np.ndarray, sr:int=16000, 
                            use_compare:bool=False) -> dict:
    """
    Function to get opensmile feature (default: eGeMAPSv02).
    """
    if use_compare:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,   # 6373 features
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    else:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,     # 88 features
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    feature_df = smile.process_signal(signal, sr)
    return feature_df.iloc[0].to_dict()