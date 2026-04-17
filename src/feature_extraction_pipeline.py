from src.features import acousticFeature
from src.features import linguisticFeature

import pandas as pd
from tqdm.auto import tqdm
import glob
from pathlib import Path

import parselmouth
import librosa
import numpy as np
from scipy.stats import skew, kurtosis

import os

from src.utils import io


# Runing on each segment and calculate statistics with extracting PRAAT features
def process_acoustic_features_praat(audio_path: str,
                                    diarization_segment_path: str,
                                    transcript_segment_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extracts and aggregates acoustic features strictly from PAR segments in csv using PRAAT-parselmouth
    """

    # Load audio file
    full_sound = parselmouth.Sound(audio_path)

    # Check matching patient id
    df_original_transcript = pd.read_csv(transcript_segment_path)
    if "files_id" in df_original_transcript.columns:
        patient_id = Path(audio_path).stem
        df_transcript = df_original_transcript[df_original_transcript["files_id"] == patient_id]

    transcript = " ".join(df_transcript["transcript"].dropna().astype(str))

    # Load the diarization csv
    df_segment = pd.read_csv(diarization_segment_path)
    par_segments = df_segment[df_segment["speaker"] == "PAR"].copy()

    segment_features_list = []

    # Iterate over each PAR segment
    for index, row in par_segments.iterrows():
        start_time = row["begin"]/1000.0
        end_time = row["end"]/1000.0

        if start_time >= end_time:
            continue

        try:
            segment_sound = full_sound.extract_part(start_time, end_time)
            # Extract feature for this segment
            intensity_attrs, _ = acousticFeature.get_intensity_attributes(
                segment_sound)
            pitch_attrs, _ = acousticFeature.get_pitch_attributes(
                segment_sound)
            jitter_attrs = acousticFeature.get_local_jitter(segment_sound)
            shimmer_attrs = acousticFeature.get_local_shimmer(segment_sound)
            spectrum_attrs, _ = acousticFeature.get_spectrum_attributes(
                segment_sound)
            formant_attrs, _ = acousticFeature.get_formant_attributes(
                segment_sound)

            # Combine to dict
            segment_features = {
                "segment_id": index,
                "start_time": start_time,
                "end_time": end_time,
                **intensity_attrs,
                **pitch_attrs,
                "jitter_local": jitter_attrs,
                "shimmer_local": shimmer_attrs,
                **spectrum_attrs,
                **formant_attrs,
            }
            segment_features_list.append(segment_features)

        except Exception as e:
            print(f"Error extracting segment {index}: {e}")
            continue

    # Convert to DataFrame
    df_segment_features = pd.DataFrame(segment_features_list)
    if df_segment_features.empty:
        statistics_features = pd.Series(dtype=float)
    else:
        numeric_df = df_segment_features.drop(
            columns=["segment_id", "start_time", "end_time"], errors="ignore"
        )

        _VAR_THRESHOLD = 1e-10

        agg_mean = numeric_df.agg("mean")
        agg_std = numeric_df.agg("std")
        agg_skew = numeric_df.agg(lambda x: skew(
            x, nan_policy="omit") if x.var() >= _VAR_THRESHOLD else 0.0)
        agg_kurt = numeric_df.agg(lambda x: kurtosis(
            x, nan_policy="omit") if x.var() >= _VAR_THRESHOLD else 0.0)

        statistics_features = pd.concat({
            "mean": agg_mean,
            "std":  agg_std,
            "skew": agg_skew,
            "kurt": agg_kurt,
        }).swaplevel().sort_index()

    return df_segment_features, statistics_features


# Extract openSMILE features
def concatenate_par_segments(audio_path,
                             par_segments,
                             sr=16000):
    full_audio, _ = librosa.load(audio_path, sr=sr)
    chunks = []
    for _, row in par_segments.iterrows():
        start = int(row["begin"] / 1000.0 * sr)
        end = int(row["end"] / 1000.0 * sr)
        if start < end:
            chunks.append(full_audio[start:end])
    return np.concatenate(chunks) if chunks else np.array([])


def process_acoustic_features_opensmile(audio_path: str,
                                        diarization_segment_path: str,
                                        transcript_segment_path: str,
                                        use_compare: bool = False) -> tuple[dict, dict]:
    """
    Extracts and aggregates acoustic features strictly from PAR segments in csv using openSMILE
    """

    df_original_transcript = pd.read_csv(transcript_segment_path)
    if "files_id" in df_original_transcript.columns:
        patient_id = Path(audio_path).stem
        df_original_transcript = df_original_transcript[
            df_original_transcript["files_id"] == patient_id
        ]

    df_segment = pd.read_csv(diarization_segment_path)
    par_segments = df_segment[df_segment["speaker"] == "PAR"].copy()

    concatenated_audio = concatenate_par_segments(audio_path, par_segments)
    opensmile_features = acousticFeature.get_opensmile_features(
        concatenated_audio, sr=16000, use_compare=use_compare)

    # Convert to DataFrame
    df_features = pd.Series(opensmile_features)

    return df_features


# Extract linguistic features
def process_linguistic_features(whisper_transcript_path: str,
                                patient_id: str,
                                lang: str = "en") -> dict:
    """
    Extract linguistic features from transcript csv
    """

    # Check matching patient id
    df_whisper_transcript = pd.read_csv(whisper_transcript_path)
    if "files_id" in df_whisper_transcript.columns:
        df_whisper_transcript = df_whisper_transcript[df_whisper_transcript["files_id"] == patient_id]

    # Extract the string from the first row, or join them if there are multiple
    whisper_transcript = " ".join(
        df_whisper_transcript["transcript"].dropna().astype(str))

    # Feature
    cttr, brunet, std_entropy, pidensity = linguisticFeature.lexical_richness(
        whisper_transcript, lang=lang)
    pos_tagged_data, polarity, subjectivity = linguisticFeature.pos_polarity_subjectivity(
        whisper_transcript, lang=lang)
    tag_count = linguisticFeature.tag_count(pos_tagged_data)
    pos_rate = linguisticFeature.evaluate_pos_rate(tag_count)
    content_density = tag_count["content_density"]
    open_class_words = tag_count["open_class_words"]
    closed_class_words = tag_count["closed_class_words"]
    disfluency_count = linguisticFeature.count_disfluency(
        whisper_transcript, lang=lang)
    person_rate, spatial_rate, temporal_rate = linguisticFeature.evaluate_deixis(
        whisper_transcript, lang=lang)
    dale_chall, flesch, coleman_liau_index, r_time, syllables = linguisticFeature.evaluate_readability(
        whisper_transcript)

    result = {
        "patient_id": patient_id,
        "lang": lang,
        "cttr": cttr,
        "brunet": brunet,
        "std_entropy": std_entropy,
        "pidensity": pidensity,
        # "pos_tagged_data": pos_tagged_data,
        "content_density": content_density,
        "open_class_words": open_class_words,
        "closed_class_words": closed_class_words,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "pos_rate": pos_rate,
        "disfluency_count": disfluency_count,
        "person_rate": person_rate,
        "spatial_rate": spatial_rate,
        "temporal_rate": temporal_rate,
        "dale_chall": dale_chall,
        "flesch": flesch,
        "coleman_liau_index": coleman_liau_index,
        "r_time": r_time,
        "syllables": syllables,
    }

    return result


# Process all features
def process_feature(audio_path: str,
                    csv_segment_path: str,
                    transcript_path: str,
                    patient_id: str,
                    diagnosis: str,
                    mmse: int,
                    lang: str = "en",
                    use_egemap02: bool = False,
                    use_compare: bool = False,
                    linguistic: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """ Process all features
    """
    if linguistic:
        processed_linguistic_feature = process_linguistic_features(
            transcript_path, patient_id, lang=lang)
    else:
        processed_linguistic_feature = {}
    if use_egemap02 or use_compare:
        df_opensmile_features = process_acoustic_features_opensmile(
            audio_path, csv_segment_path, transcript_path, use_compare=use_compare)
    else:
        _, df_praat_features = process_acoustic_features_praat(
            audio_path, csv_segment_path, transcript_path)

    # patient_id and diagnosis always appear as first columns in both DataFrames
    meta = {"patient_id": patient_id, "diagnosis": diagnosis, "mmse": mmse}

    # Flatten linguistic features
    if linguistic:
        flat_ling = {**meta}
        for k, v in processed_linguistic_feature.items():
            if k == "patient_id":       # already added via meta, skip duplicate
                continue
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_ling[f"{k}_{sub_k}"] = sub_v
            else:
                flat_ling[k] = v
    else:
        flat_ling = {}

    # Flatten acoustic features
    flat_acoustic = {**meta}
    if (use_egemap02 or use_compare):
        for feat, val in df_opensmile_features.items():
            flat_acoustic[feat] = val
    else:
        for (feat, stat), val in df_praat_features.items():
            flat_acoustic[f"{feat}_{stat}"] = val

    return pd.DataFrame([flat_ling]), pd.DataFrame([flat_acoustic])


# Extract all features
def extract_features(output_dir: str,
                     whisper_transcript_path: str,
                     use_egemap02: bool = False,
                     use_compare: bool = False,
                     linguistic: bool = False,
                     data_type: str = "train",
                     save_csv: bool = True) -> None:

    # Extract acoustic feature
    if use_egemap02:
        output_acoustic_file = Path(output_dir) / \
            f"adresso_egemaps_{data_type}.csv"
    elif use_compare:
        output_acoustic_file = Path(output_dir) / \
            f"adresso_compare_{data_type}.csv"
    else:
        output_acoustic_file = Path(output_dir) / \
            f"adresso_praat_{data_type}.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Linguistic features are independent of acoustic method
    if linguistic:
        output_linguistic_file = Path(
            output_dir) / f"adresso_linguistic_{data_type}.csv"

    transcript_files = glob.glob(
        whisper_transcript_path + f"/adresso_transcripts_{data_type}.csv")
    transcript_files = transcript_files[0]

    # Sample information and data
    df_sample_info = pd.read_csv(transcript_files)
    patient_id = df_sample_info["files_id"]
    audio_path = df_sample_info["audio_path"]
    diagnosis_list = df_sample_info["diagnosis"]
    segment_path_list = df_sample_info["segment_path"]
    mmse_list = df_sample_info["mmse_score"]

    df_ling_list = []
    df_acoustic_list = []

    if linguistic:
        print("Extracting linguistic features...")
    if use_egemap02:
        print("Extracting acoustic features using eGeMAPS...")
    elif use_compare:
        print("Extracting acoustic features using ComParE...")
    else:
        print("Extracting acoustic features using Praat...")

    for i in tqdm(range(len(df_sample_info))):
        patient = patient_id[i]
        diag = diagnosis_list[i]
        segment_file = segment_path_list[i]
        mmse = mmse_list[i]

        # Eliminate the samples without interviewee saying
        if not os.path.exists(segment_file):
            continue

        df_segment = pd.read_csv(segment_file)
        if "PAR" not in df_segment["speaker"].values:
            continue

        df_ling_row, df_acoustic_row = process_feature(
            audio_path[i], segment_file, transcript_files, patient, diag, mmse,
            lang="en",
            use_egemap02=use_egemap02,
            use_compare=use_compare,
            linguistic=linguistic
        )

        if linguistic:
            df_ling_list.append(df_ling_row)
        df_acoustic_list.append(df_acoustic_row)

    if save_csv:
        if linguistic and df_ling_list:
            df_linguistic = pd.concat(df_ling_list, ignore_index=True)
            df_linguistic.to_csv(output_linguistic_file, index=False)

        if df_acoustic_list:
            df_acoustic = pd.concat(df_acoustic_list, ignore_index=True)
            df_acoustic.to_csv(output_acoustic_file, index=False)


def feature_extraction_pipeline() -> None:
    path_config = io.load_yaml("src/config/path.yaml")
    whisper_transcript_path = path_config["TRANSCRIPT_PATH"]

    # Data extraction TRAIN
    # Extract train linguistic features and praat feature
    extract_features(output_dir=path_config["OUTPUT_TRADITIONAL_FEATURE_PATH"],
                     whisper_transcript_path=whisper_transcript_path,
                     data_type="train",
                     use_egemap02=False, use_compare=False, linguistic=True)
    # Extract train egeMAP02 features
    extract_features(output_dir=path_config["OUTPUT_TRADITIONAL_FEATURE_PATH"],
                     whisper_transcript_path=whisper_transcript_path,
                     data_type="train",
                     use_egemap02=True, use_compare=False, linguistic=False)
    # Extract train ComParE features
    extract_features(output_dir=path_config["OUTPUT_TRADITIONAL_FEATURE_PATH"],
                     whisper_transcript_path=whisper_transcript_path,
                     data_type="train",
                     use_egemap02=False, use_compare=True, linguistic=False)
    print("finish feature extraction train set")

    # Data extraction TEST
    # Extract test linguistic features and praat feature
    extract_features(output_dir=path_config["OUTPUT_TRADITIONAL_FEATURE_PATH"],
                     whisper_transcript_path=whisper_transcript_path,
                     data_type="test",
                     use_egemap02=False, use_compare=False, linguistic=True)
    # Extract test egeMAP02 features
    extract_features(output_dir=path_config["OUTPUT_TRADITIONAL_FEATURE_PATH"],
                     whisper_transcript_path=whisper_transcript_path,
                     data_type="test",
                     use_egemap02=True, use_compare=False, linguistic=False)
    # Extract test ComParE features
    extract_features(output_dir=path_config["OUTPUT_TRADITIONAL_FEATURE_PATH"],
                     whisper_transcript_path=whisper_transcript_path,
                     data_type="test",
                     use_egemap02=False, use_compare=True, linguistic=False)
    print("finish feature extraction test set")
