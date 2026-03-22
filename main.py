import os
import src.utils as utils
from pathlib import Path
import pandas as pd
import torch


if __name__ == "__main__":
    BASE_PATH = "/mnt/data_lab513/ducvu/ADReSSo/ADReSSo-feature-extration"

    OUTPUT_FEATURE_PATH = f"{BASE_PATH}/output/features"

    # Train paths
    CSV_SEGMENT_TRAIN_PATH = f"{BASE_PATH}/data/diagnosis/train/segmentation"
    AUDIO_TRAIN_PATH = f"{BASE_PATH}/data/diagnosis/train/audio"
    MMSE_DIAG_TRAIN_PATH = f"{BASE_PATH}/data/diagnosis/train/adresso-train-mmse-scores.csv"

    # Test paths
    CSV_SEGMENT_TEST_PATH = f"{BASE_PATH}/data/diagnosis/test-dist/segmentation"
    AUDIO_TEST_PATH = f"{BASE_PATH}/data/diagnosis/test-dist/audio"
    MMSE_DIAG_TEST_PATH = f"{BASE_PATH}/data/diagnosis/test-dist/adresso-test-mmse-scores.csv"

    # Extract transcripts
    TRANSCRIPT_PATH = f"{BASE_PATH}/output/transcripts"
    # Model setup
    MODEL_NAME = "openai/whisper-large-v3"
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Get files
    audio_train_files = utils.get_files(AUDIO_TRAIN_PATH, data_type="train")
    audio_test_files = utils.get_files(AUDIO_TEST_PATH, data_type="test")

    # df_train_transcripts = utils.transcribe_audio_files(audio_train_files, MMSE_DIAG_TRAIN_PATH, 
    #                                                     CSV_SEGMENT_TRAIN_PATH, data_type="train", 
    #                                                     multipleGPU=False, model_name=MODEL_NAME, 
    #                                                     batch_size=BATCH_SIZE, device=DEVICE, 
    #                                                     output_path=TRANSCRIPT_PATH)
    df_test_transcripts = utils.transcribe_audio_files(audio_test_files, MMSE_DIAG_TEST_PATH, 
                                                        CSV_SEGMENT_TEST_PATH, data_type="test", 
                                                        multipleGPU=False, model_name=MODEL_NAME, 
                                                        batch_size=BATCH_SIZE, device=DEVICE, 
                                                        output_path=TRANSCRIPT_PATH)







    # # Extract linguistic and PRAAT features
    # utils.feature_extraction(OUTPUT_FEATURE_PATH, TRANSCRIPT_PATH, CSV_SEGMENT_PATH,
    #                             use_egemap02=False, use_compare=False, linguistic=True)
    # # Extract eGeMAPS features
    # utils.feature_extraction(OUTPUT_FEATURE_PATH, TRANSCRIPT_PATH, CSV_SEGMENT_PATH,
    #                             use_egemap02=True, use_compare=False, linguistic=False)
    # # Extract ComParE features
    # utils.feature_extraction(OUTPUT_FEATURE_PATH, TRANSCRIPT_PATH, CSV_SEGMENT_PATH,
    #                             use_egemap02=False, use_compare=True, linguistic=False)

    # # Load all features
    # compare_features = pd.read_csv(f"{OUTPUT_FEATURE_PATH}/adresso_features_compare.csv")
    # print(f"Compare features: {compare_features.shape}")
    # egemap_features = pd.read_csv(f"{OUTPUT_FEATURE_PATH}/adresso_features_egemaps.csv")
    # print(f"eGeMAPS features: {egemap_features.shape}")
    # praat_features = pd.read_csv(f"{OUTPUT_FEATURE_PATH}/adresso_features_praat.csv")
    # print(f"PRAAT features: {praat_features.shape}")
    # linguistic_features = pd.read_csv(f"{OUTPUT_FEATURE_PATH}/adresso_features_linguistic.csv")
    # print(f"Linguistic features: {linguistic_features.shape}")

    # X = utils.get_audio_files()