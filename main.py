from src.utils import io
from src import feature_extraction_pipeline
from src import transcription_pipeline
from src import model_feature_pipline
import glob
from pathlib import Path

import pandas as pd


def main_traditional_approach(transcript: bool = False,
                              feature: bool = False,
                              classification_model: bool = False,
                              early_fusion: bool = False,
                              feature_selection: bool = False) -> None:

    # Transcribe audio files
    if transcript:
        transcription_pipeline.transcript_pipeline()
        print(f"finish transcript train and test set")

    if feature:
        feature_extraction_pipeline.feature_extraction_pipeline()
        print(f"finish feature extraction train and test set")
        path_config = io.load_yaml("src/config/path.yaml")
        pkl_path = path_config["PKL_TRADITIONAL_PATH"]
        feature_path = path_config["OUTPUT_TRADITIONAL_FEATURE_PATH"]
        Path(pkl_path).mkdir(parents=True, exist_ok=True)
        feature_list_files = glob.glob(feature_path + "/*.csv")
        for feature_file in feature_list_files:
            feature_file_name = Path(feature_file).stem
            pkl_file_path = pkl_path + "/" + feature_file_name + ".pkl"
            io.csv_to_pkl(csv_path=feature_file, pkl_path=pkl_file_path)
        print(f"finish converting to pkl train and test set")

    if classification_model:
        # Define test configurations
        # Run model pipeline with selected feature selection setting
        # Without correlation_threshold
        tests_hybrid_pca_without_corr = {
            "hybrid": ["compare", "egemaps", "linguistic", "praat"],
            "pca": ["compare", "egemaps", "linguistic", "praat"]
        }
        model_feature_pipline.model_pipeline(tests_hybrid_pca_without_corr,
                                             early_fusion=early_fusion,
                                             feature_selection=feature_selection,
                                             correlation_threshold=0.0,
                                             output_csv_name="hybrid_pca_withtout_corr")

        # Correlation_threshold
        tests_hybrid_corr = {
            "hybrid": ["compare", "egemaps", "linguistic", "praat"],
        }
        model_feature_pipline.model_pipeline(tests_hybrid_corr,
                                             early_fusion=early_fusion,
                                             feature_selection=feature_selection,
                                             correlation_threshold=0.9,
                                             output_csv_name="hybrid_corr")
        # Raw feature
        tests_raw_features = {
            "": ["compare", "egemaps", "linguistic", "praat"],
        }
        model_feature_pipline.model_pipeline(tests_raw_features,
                                             early_fusion=early_fusion,
                                             feature_selection=False,
                                             correlation_threshold=None,
                                             output_csv_name="raw")

        # Merged feature


if __name__ == "__main__":

    main_traditional_approach(transcript=False,
                              feature=False,
                              classification_model=True, early_fusion=False, feature_selection=True
                              )
