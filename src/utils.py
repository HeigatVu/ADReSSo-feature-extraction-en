import os
from typing import Dict, List
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
import glob
import torch
from transformers import pipeline

from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()

from src import acousticFeature
from src import linguisticFeature

import parselmouth
import librosa
import numpy as np
import soundfile as sf
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split

def get_files(audio_path:str, 
                    data_type:str="train") -> dict[str, List[str]]:
    """Load all audio files from ADReSSo structure
    """
    if data_type == "train":
        audio_files = {
            "ad": sorted((Path(audio_path) / "ad").glob('*.*')),
            "cn": sorted((Path(audio_path) / "cn").glob('*.*')),
        }
    elif data_type == "test":
        audio_files = {
            "audio": sorted((Path(audio_path)).glob('*.*')),
        }
    else:
        raise ValueError("Invalid data type. Must be 'train' or 'test'.")
    
    return audio_files


# Transcript
def transcribe_audio_files(audio_files:Dict[str, List[Path]], 
                            mmse_diagnosis_path:str,
                            csv_segment_path:str,
                            data_type:str="train",
                            multipleGPU: bool = False,
                            model_name:str = "openai/whisper-large-v3", 
                            batch_size:int = 8,
                            device:str = "cuda" if torch.cuda.is_available() else "cpu",
                            output_path:str = None,
                            hf_token:str = None) -> pd.DataFrame:
    """ Transcribe audio files without diarization
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    torch_type=torch.float16 if torch.cuda.is_available() else torch.float32
    if not multipleGPU:
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            batch_size=batch_size,
            torch_dtype=torch_type,
            token=token,
            model_kwargs={
                "attn_implementation": "sdpa"  # Scaled dot-product attention (faster)
            }
        )

        transcriber.model.config.forced_decoder_ids = transcriber.tokenizer.get_decoder_prompt_ids(
            language="english",
            task="transcribe"
        )

    else:
        transcriber = pipeline(
            task="automatic-speech-recognition",
            batch_size=batch_size,
            model=model_name,
            device=device,
            torch_dtype=torch_type,
            device_map="auto", # Using this to multiGPU
            token=token,
            generate_kwargs={
                "language": "english"
            },
            model_kwargs={
                "attn_implementation": "sdpa"  # Scaled dot-product attention (faster)
            }
        )

    results = []
    df_mmse = pd.read_csv(mmse_diagnosis_path)
    df_mmse.columns = [c.strip().strip('"').strip() for c in df_mmse.columns]

    if data_type == "train":
        for diagnosis, files in audio_files.items():
            for audio_file in tqdm(files, desc=f"{diagnosis.upper()}"):
                output = transcriber(
                    str(audio_file),
                    return_timestamps=True,
                    generate_kwargs={
                        "task": "transcribe",
                        "language": "en",
                        "return_timestamps": True,
                        "num_beams": 5,
                    }
                )

                # Handle different output formats
                if isinstance(output, dict):
                    if "text" in output:
                        transcript = output["text"].strip()
                    elif "chunks" in output:
                        transcript = " ".join([chunk["text"] for chunk in output["chunks"]]).strip()
                    else:
                        transcript = ""
                else:
                    transcript = str(output).strip()

                patient_row = df_mmse[df_mmse["adressfname"] == audio_file.stem]
                segment_path = csv_segment_path + f"/{diagnosis}/{audio_file.stem}.csv"

                results.append({
                    "files_id": audio_file.stem,
                    "mmse_score": patient_row["mmse"].values[0],
                    "audio_path": str(audio_file),
                    "diagnosis": diagnosis,
                    "segment_path": str(segment_path),
                    "transcript": transcript,

                    })
        output_file = Path(output_path) / "adresso_transcripts_train.csv"
        pd.DataFrame(results).to_csv(output_file, index=False)

    if data_type == "test":
        for audio_file in tqdm(audio_files.get("audio", []), desc="TEST"):
            output = transcriber(
                str(audio_file),
                return_timestamps=True,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "en",
                    "return_timestamps": True,
                    "num_beams": 5,
                }
            )

            # Handle different output formats
            if isinstance(output, dict):
                if "text" in output:
                    transcript = output["text"].strip()
                elif "chunks" in output:
                    transcript = " ".join([chunk["text"] for chunk in output["chunks"]]).strip()
                else:
                    transcript = ""
            else:
                transcript = str(output).strip()

            patient_row = df_mmse[df_mmse["adressfname"] == audio_file.stem]
            segment_path = csv_segment_path + f"/{audio_file.stem}.csv"
            if patient_row["dx"].values[0] == "Control":
                diagnosis = "cn"
            if patient_row["dx"].values[0] == "ProbableAD":
                diagnosis = "ad"

            results.append({
                "files_id": audio_file.stem,
                "mmse_score": patient_row["mmse"].values[0],
                "audio_path": str(audio_file),
                "diagnosis": diagnosis,
                "segment_path": str(segment_path),
                "transcript": transcript,
                })
        output_file = Path(output_path) / "adresso_transcripts_test.csv"
        pd.DataFrame(results).to_csv(output_file, index=False)

    return f"Done {data_type}"


######################################### TRADITIONAL APROACH #########################################
# Runing on each segment and calculate statistics with extracting PRAAT features
def process_acoustic_features_praat(audio_path:str, 
                                    diarization_segment_path:str, 
                                    transcript_segment_path:str) -> tuple[pd.DataFrame, pd.Series]:
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
            intensity_attrs, _ = acousticFeature.get_intensity_attributes(segment_sound)
            pitch_attrs, _ = acousticFeature.get_pitch_attributes(segment_sound)
            jitter_attrs = acousticFeature.get_local_jitter(segment_sound)
            shimmer_attrs = acousticFeature.get_local_shimmer(segment_sound)
            spectrum_attrs, _ = acousticFeature.get_spectrum_attributes(segment_sound)
            formant_attrs, _ = acousticFeature.get_formant_attributes(segment_sound)

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
        agg_std  = numeric_df.agg("std")
        agg_skew = numeric_df.agg(lambda x: skew(x, nan_policy="omit") if x.var() >= _VAR_THRESHOLD else 0.0)
        agg_kurt = numeric_df.agg(lambda x: kurtosis(x, nan_policy="omit") if x.var() >= _VAR_THRESHOLD else 0.0)

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
        end   = int(row["end"]   / 1000.0 * sr)
        if start < end:
            chunks.append(full_audio[start:end])
    return np.concatenate(chunks) if chunks else np.array([])

def process_acoustic_features_opensmile(audio_path:str, 
                                        diarization_segment_path:str, 
                                        transcript_segment_path:str, 
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
    opensmile_features = acousticFeature.get_opensmile_features(concatenated_audio, sr=16000, use_compare=use_compare)

    # Convert to DataFrame
    df_features = pd.Series(opensmile_features)

    return df_features


# Extract linguistic features
def process_linguistic_features(whisper_transcript_path:str, 
                                patient_id:str, 
                                lang:str="en") -> dict:
    """
    Extract linguistic features from transcript csv
    """

    # Check matching patient id
    df_whisper_transcript = pd.read_csv(whisper_transcript_path)
    if "files_id" in df_whisper_transcript.columns:
        df_whisper_transcript = df_whisper_transcript[df_whisper_transcript["files_id"] == patient_id]
        
    # Extract the string from the first row, or join them if there are multiple
    whisper_transcript = " ".join(df_whisper_transcript["transcript"].dropna().astype(str))

    # Feature
    cttr, brunet, std_entropy, pidensity = linguisticFeature.lexical_richness(whisper_transcript, lang=lang)
    pos_tagged_data, polarity, subjectivity = linguisticFeature.pos_polarity_subjectivity(whisper_transcript, lang=lang)
    tag_count = linguisticFeature.tag_count(pos_tagged_data)
    pos_rate = linguisticFeature.evaluate_pos_rate(tag_count)
    content_density = tag_count["content_density"]
    open_class_words = tag_count["open_class_words"]
    closed_class_words = tag_count["closed_class_words"]
    disfluency_count = linguisticFeature.count_disfluency(whisper_transcript, lang=lang)
    person_rate, spatial_rate, temporal_rate = linguisticFeature.evaluate_deixis(whisper_transcript, lang=lang)
    dale_chall, flesch, coleman_liau_index, r_time, syllables = linguisticFeature.evaluate_readability(whisper_transcript)


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
def process_feature(audio_path:str, 
                    csv_segment_path:str, 
                    transcript_path:str, 
                    patient_id:str, 
                    diagnosis:str, 
                    lang:str="en",
                    use_egemap02:bool=False, 
                    use_compare:bool=False, 
                    linguistic:bool=True) -> tuple[pd.DataFrame, pd.Series]:
    """ Process all features
    """
    if linguistic:
        processed_linguistic_feature = process_linguistic_features(transcript_path, patient_id, lang=lang)
    else:
        processed_linguistic_feature = {}
    if use_egemap02 or use_compare:
        df_opensmile_features = process_acoustic_features_opensmile(
            audio_path, csv_segment_path, transcript_path, use_compare=use_compare)
    else:
        _, df_praat_features = process_acoustic_features_praat(
            audio_path, csv_segment_path, transcript_path)
    
    # patient_id and diagnosis always appear as first columns in both DataFrames
    meta = {"patient_id": patient_id, "diagnosis": diagnosis}

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
def feature_extraction(output_dir: str, 
                        whisper_transcription_path:str, 
                        use_egemap02:bool=False, 
                        use_compare:bool=False, 
                        linguistic:bool=False, 
                        data_type:str="train",
                        save_csv:bool=True) -> str:
    
    # Extract acoustic feature
    if use_egemap02:
        output_acoustic_file = Path(output_dir) / "adresso_features_egemaps.csv"
    elif use_compare:
        output_acoustic_file = Path(output_dir) / "adresso_features_compare.csv"
    else:
        output_acoustic_file = Path(output_dir) / "adresso_features_praat.csv"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Linguistic features are independent of acoustic method
    if linguistic:
        output_linguistic_file = Path(output_dir) / "adresso_features_linguistic.csv"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    transcript_files = glob.glob(whisper_transcription_path + f"/adresso_transcripts_{data_type}.csv")

    # Sample information and data
    df_sample_info = pd.read_csv(transcript_files)
    patient_id = df_sample_info["files_id"]
    audio_path = df_sample_info["audio_path"]
    diagnosis_list = df_sample_info["diagnosis"]
    segment_path_list = df_sample_info["segment_path"]
    
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
        
        # Eliminate the samples without interviewee saying
        if not os.path.exists(segment_file):
            continue
            
        df_segment = pd.read_csv(segment_file)
        if "PAR" not in df_segment["speaker"].values:
            continue
            
        df_ling_row, df_acoustic_row = process_feature(
            audio_path[i], segment_file, transcript_files, patient, diag,
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

    return df_linguistic, df_acoustic


# Split data into train, validation, and test sets
def train_val_test_split(df_train:pd.DataFrame, 
                        test_size:float=0.2, 
                        random_state:int=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Split data into train and validation datasets
    """
    
    df_x_train, df_x_val, df_y_train, df_y_val = train_test_split(df_train, test_size=test_size, 
                                                                random_state=random_state)

    return df_x_train, df_x_val, df_y_train, df_y_val

