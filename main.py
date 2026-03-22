from src import utils
from src import feature_extraction_pipeline
from src import transcription_pipeline

import torch

def main_traditional_approach(transcript:bool=False, 
                            feature:bool=False, 
                            feature_selection:bool=False) -> str:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Transcribe audio files
    if transcript:
        # Get files
        path_config = utils.load_yaml("src/config/path.yaml")
        audio_train_files = utils.get_files(path_config["train"]["AUDIO_TRAIN_PATH"], data_type="train")
        audio_test_files = utils.get_files(path_config["test"]["AUDIO_TEST_PATH"], data_type="test")

        model_config = utils.load_yaml("src/config/model.yaml")
        transcription_pipeline.transcribe_audio_files(audio_train_files, 
                                    path_config["train"]["MMSE_DIAG_TRAIN_PATH"], 
                                    path_config["train"]["CSV_SEGMENT_TRAIN_PATH"], 
                                    data_type="train", 
                                    multipleGPU=model_config["whisper"]["MULTIPLE_GPU"], 
                                    model_name=model_config["whisper"]["MODEL_NAME"], 
                                    batch_size=model_config["whisper"]["BATCH_SIZE"], 
                                    device=DEVICE, 
                                    output_path=path_config["TRANSCRIPT_PATH"])
        transcription_pipeline.transcribe_audio_files(audio_test_files, 
                                    path_config["test"]["MMSE_DIAG_TEST_PATH"], 
                                    path_config["test"]["CSV_SEGMENT_TEST_PATH"], 
                                    data_type="test", 
                                    multipleGPU=model_config["whisper"]["MULTIPLE_GPU"], 
                                    model_name=model_config["whisper"]["MODEL_NAME"], 
                                    batch_size=model_config["whisper"]["BATCH_SIZE"], 
                                    device=DEVICE, 
                                    output_path=path_config["TRANSCRIPT_PATH"])
        return f"finish transcript train and test set"
    
    if feature:
        path_config = utils.load_yaml("src/config/path.yaml")
        whisper_transcript_path = path_config["TRANSCRIPT_PATH"]

        # Data extraction TRAIN
        # Extract train linguistic features and praat feature
        feature_extraction_pipeline.extract_features(output_dir=path_config["OUTPUT_FEATURE_PATH"], 
                                whisper_transcript_path=whisper_transcript_path,
                                data_type="train", 
                                use_egemap02=False, use_compare=False, linguistic=True)
        # Extract train egeMAP02 features
        feature_extraction_pipeline.extract_features(output_dir=path_config["OUTPUT_FEATURE_PATH"], 
                                whisper_transcript_path=whisper_transcript_path,
                                data_type="train", 
                                use_egemap02=True, use_compare=False, linguistic=False)
        # Extract train ComParE features
        feature_extraction_pipeline.extract_features(output_dir=path_config["OUTPUT_FEATURE_PATH"], 
                                whisper_transcript_path=whisper_transcript_path,
                                data_type="train", 
                                use_egemap02=False, use_compare=True, linguistic=False)
        print("finish feature extraction train set")

        # Data extraction TEST
        # Extract test linguistic features and praat feature
        feature_extraction_pipeline.extract_features(output_dir=path_config["OUTPUT_FEATURE_PATH"], 
                                whisper_transcript_path=whisper_transcript_path,
                                data_type="test", 
                                use_egemap02=False, use_compare=False, linguistic=True)
        # Extract test egeMAP02 features
        feature_extraction_pipeline.extract_features(output_dir=path_config["OUTPUT_FEATURE_PATH"], 
                                whisper_transcript_path=whisper_transcript_path,
                                data_type="test", 
                                use_egemap02=True, use_compare=False, linguistic=False)
        # Extract test ComParE features
        feature_extraction_pipeline.extract_features(output_dir=path_config["OUTPUT_FEATURE_PATH"], 
                                whisper_transcript_path=whisper_transcript_path,
                                data_type="test", 
                                use_egemap02=False, use_compare=True, linguistic=False)
        print("finish feature extraction test set")
        return f"finish feature extraction train and test set"

    if feature_selection:
        pass
        return f"finish feature selection on train dataset"


if __name__ == "__main__":

    main_traditional_approach(transcript=False, feature=True)