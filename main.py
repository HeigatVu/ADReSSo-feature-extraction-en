from src.utils import io
from src import feature_extraction_pipeline
from src import transcription_pipeline
from src import model_feature_pipeline
import glob
from pathlib import Path
from tqdm import tqdm

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
        print(f"Models for single feature with PCA and without correlation")
        tests_hybrid_pca_without_corr = {
            "hybrid": ["compare", "egemaps", "linguistic", "praat"],
            "pca": ["compare", "egemaps", "linguistic", "praat"]
        }
        model_feature_pipeline.model_pipeline_one_feature(tests_hybrid_pca_without_corr,
                                                         feature_selection=feature_selection,
                                                         correlation_threshold=0.0,
                                                         output_csv_name="hybrid_pca_withtout_corr")

        # Correlation_threshold
        print(f"Models for single feature with correlation")
        tests_hybrid_corr = {
            "hybrid": ["compare", "egemaps", "linguistic", "praat"],
        }
        model_feature_pipeline.model_pipeline_one_feature(tests_hybrid_corr,
                                                         feature_selection=feature_selection,
                                                         correlation_threshold=0.9,
                                                         output_csv_name="hybrid_corr")
        # Raw feature
        print(f"Models for single raw feature")
        tests_raw_features = {
            "": ["compare", "egemaps", "linguistic", "praat"],
        }
        model_feature_pipeline.model_pipeline_one_feature(tests_raw_features,
                                                         feature_selection=False,
                                                         correlation_threshold=None,
                                                         output_csv_name="raw")

        # Merged feature
        acoustic_list = ["compare", "egemaps", "praat"]
        print(f"Early fusion model without correlation")
        model_feature_pipeline.early_fusion_pipeline(acoustic_list=acoustic_list,
                                                    linguistic_type="linguistic",
                                                    strategy="hybrid",
                                                    output_csv_name="results_early_merged_hybrid_withoutCorr_model")
        model_feature_pipeline.early_fusion_pipeline(acoustic_list=acoustic_list,
                                                    linguistic_type="linguistic",
                                                    strategy="pca",
                                                    output_csv_name="results_early_merged_pca_model")

        print(f"Early fusion model with correlation")
        model_feature_pipeline.early_fusion_pipeline(acoustic_list=acoustic_list,
                                                    linguistic_type="linguistic",
                                                    strategy="hybrid",
                                                    correlation_threshold=0.9,
                                                    output_csv_name="results_early_merged_hybrid_Corr_model")

        print(f"Late fusion model without correlation")
        model_feature_pipeline.late_fusion_pipeline(acoustic_list=acoustic_list,
                                                   linguistic_type="linguistic",
                                                   strategy="hybrid",
                                                   output_csv_name="results_late_merged_hybrid_withoutCorr_model")
        model_feature_pipeline.late_fusion_pipeline(acoustic_list=acoustic_list,
                                                   linguistic_type="linguistic",
                                                   strategy="pca",
                                                   output_csv_name="results_late_merged_pca_model")

        print(f"Late fusion model with correlation")
        model_feature_pipeline.late_fusion_pipeline(acoustic_list=acoustic_list,
                                                   linguistic_type="linguistic",
                                                   strategy="hybrid",
                                                   correlation_threshold=0.9,
                                                   output_csv_name="results_late_merged_hybrid_Corr_model")


if __name__ == "__main__":

    main_traditional_approach(transcript=False,
                              feature=False,
                              classification_model=True, early_fusion=False, feature_selection=True
                              )
