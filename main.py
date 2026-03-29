from src.utils import io
from src import feature_extraction_pipeline
from src import transcription_pipeline
from src import model_feature_pipline
from src.utils import helperFn
import glob
from pathlib import Path

import pandas as pd


def main_traditional_approach(transcript:bool=False, 
                            feature:bool=False, 
                            classification_model:bool=False,
                            early_fusion:bool=False) -> str:

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
        tests = {
            "hybrid": ["compare", "egemaps", "linguistic", "praat"],
            "pca": ["compare", "egemaps", "linguistic", "praat"]
        }
        # Raw feature model
        model_feature_pipline.model_pipeline(tests, 
                                            early_fusion=early_fusion, 
                                            feature_selection=False, 
                                            threshold=threshold)

        # Selected feature model
        model_feature_pipline.model_pipeline(tests, 
                                            early_fusion=early_fusion, 
                                            feature_selection=True, 
                                            threshold=threshold)

        # # Selected feature model removing correlated features threshold 0.9
        # model_feature_pipline.model_pipeline(tests, 
        #                                     early_fusion=early_fusion, 
        #                                     feature_selection=True, 
        #                                     threshold=0.9)

        # Merged feature

if __name__ == "__main__":

    # main_traditional_approach(transcript=False, 
    #                         feature=False, 
    #                         classification_model=True, early_fusion=False, feature_selection=True
    #                         )

    csv_ling_path = "/home/heigatvu/MyFile/my-project/ADReSSo-feature-extraction/output/features/adresso_linguistic_train.csv"
    csv_egemaps_path = "/home/heigatvu/MyFile/my-project/ADReSSo-feature-extraction/output/features/adresso_egemaps_train.csv"
    csv_compare_path = "/home/heigatvu/MyFile/my-project/ADReSSo-feature-extraction/output/features/adresso_compare_train.csv"
    csv_praat_path = "/home/heigatvu/MyFile/my-project/ADReSSo-feature-extraction/output/features/adresso_praat_train.csv"

    df_ling = pd.read_csv(csv_ling_path)
    print(df_ling.shape)
    df_egemaps = pd.read_csv(csv_egemaps_path)
    print(df_egemaps.shape)
    df_compare = pd.read_csv(csv_compare_path)
    print(df_compare.shape)
    df_praat = pd.read_csv(csv_praat_path)
    print(df_praat.shape)

    pkl_ling_path = "/home/heigatvu/MyFile/my-project/ADReSSo-feature-extraction/output/pkl/traditionalFeatures/adresso_linguistic_train.pkl"
    pkl_egemaps_path = "/home/heigatvu/MyFile/my-project/ADReSSo-feature-extraction/output/pkl/traditionalFeatures/adresso_egemaps_train.pkl"
    pkl_compare_path = "/home/heigatvu/MyFile/my-project/ADReSSo-feature-extraction/output/pkl/traditionalFeatures/adresso_compare_train.pkl"

    ling_df = io.load_data(pkl_path=pkl_ling_path, meta_data=True, df_csv=df_ling)
    egemaps_df = io.load_data(pkl_path=pkl_egemaps_path, meta_data=True, df_csv=df_egemaps)
    compare_df = io.load_data(pkl_path=pkl_compare_path, meta_data=True, df_csv=df_compare)
    
    X_fused, y_ling = helperFn.fused_feature(ling_df, egemaps_df, compare_df)
    print(X_fused.head())