from src.utils import io
from src import feature_extraction_pipeline
from src import transcription_pipeline
from src import model_feature_selection_pipline
import glob
from pathlib import Path


def main_traditional_approach(transcript:bool=False, 
                            feature:bool=False, 
                            classification_model:bool=False,
                            feature_selection:bool=False,
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
        model_feature_selection_pipline.model_pipeline(tests, early_fusion=early_fusion, feature_selection=feature_selection)
                

if __name__ == "__main__":

    main_traditional_approach(transcript=False, 
                            feature=False, 
                            classification_model=True, early_fusion=False, feature_selection=True
                            )