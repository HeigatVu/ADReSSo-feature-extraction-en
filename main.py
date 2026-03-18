import os
import src.utils as utils
import glob
import pandas as pd

def process_feature(audio_path:str, csv_segment_path:str, transcript_path:str, patient_id:str, lang:str="en") -> None:
    processed_linguistic_feature = utils.process_linguistic_features(transcript_path, patient_id, lang=lang)
    processed_acoustic_feature = utils.process_acoustic_features(audio_path, csv_segment_path)

    # print(f"linguistic feature: \n{processed_linguistic_feature}")
    # print(f"acoustic feature: \n{processed_acoustic_feature}")

    



if __name__ == "__main__":
    BASE_PATH = "/mnt/data_lab513/ducvu/ADReSSo/ADReSSo-feature-extration"
    TRANSCRIPT_PATH = f"{BASE_PATH}/output/transcripts"
    AUDIO_PATH = "/mnt/data_lab513/ducvu/ADReSSo/ADReSSo-feature-extration/data/diagnosis/train/audio/ad/adrso024.wav"
    CSV_PATH = "/mnt/data_lab513/ducvu/ADReSSo/ADReSSo-feature-extration/data/diagnosis/train/segmentation/ad/adrso024.csv"

    transcript_files = glob.glob(TRANSCRIPT_PATH + "/*.csv")[0]
    df = pd.read_csv(transcript_files)
    test_transcript = df["transcript"][0]
    test_patient_id = df["files_id"][0]

    process_feature(AUDIO_PATH, CSV_PATH, test_transcript, test_patient_id, lang="en")
