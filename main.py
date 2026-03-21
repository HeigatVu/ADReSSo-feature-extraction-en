import os
import src.utils as utils
from pathlib import Path

if __name__ == "__main__":
    BASE_PATH = "/mnt/data_lab513/ducvu/ADReSSo/ADReSSo-feature-extration"
    TRANSCRIPT_PATH = f"{BASE_PATH}/output/transcripts"
    CSV_SEGMENT_PATH = f"{BASE_PATH}/data/diagnosis/train/segmentation"
    OUTPUT_FEATURE_PATH = f"{BASE_PATH}/output/features"

    # Extract linguistic and PRAAT features
    utils.feature_extraction(OUTPUT_FEATURE_PATH, TRANSCRIPT_PATH, CSV_SEGMENT_PATH,
                                use_egemap02=False, use_compare=False, linguistic=True)
    # Extract eGeMAPS features
    utils.feature_extraction(OUTPUT_FEATURE_PATH, TRANSCRIPT_PATH, CSV_SEGMENT_PATH,
                                use_egemap02=True, use_compare=False, linguistic=False)
    # Extract ComParE features
    utils.feature_extraction(OUTPUT_FEATURE_PATH, TRANSCRIPT_PATH, CSV_SEGMENT_PATH,
                                use_egemap02=False, use_compare=True, linguistic=False)