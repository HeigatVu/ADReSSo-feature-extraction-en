from src.utils import io
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers import pipeline
import torch

from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()


# Transcript
def transcribe_audio_files(audio_files: Dict[str, List[Path]],
                           mmse_diagnosis_path: str,
                           csv_segment_path: str,
                           data_type: str = "train",
                           multipleGPU: bool = False,
                           model_name: str = "openai/whisper-large-v3",
                           batch_size: int = 8,
                           device: str = "cuda" if torch.cuda.is_available() else "cpu",
                           output_path: str = None,
                           hf_token: str = None) -> pd.DataFrame:
    """ Transcribe audio files without diarization
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    torch_type = torch.float16 if torch.cuda.is_available() else torch.float32
    if not multipleGPU:
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            batch_size=batch_size,
            dtype=torch_type,
            token=token,
            model_kwargs={
                # Scaled dot-product attention (faster)
                "attn_implementation": "sdpa"
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
            dtype=torch_type,
            device_map="auto",  # Using this to multiGPU
            token=token,
            generate_kwargs={
                "language": "english"
            },
            model_kwargs={
                # Scaled dot-product attention (faster)
                "attn_implementation": "sdpa"
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
                        transcript = " ".join(
                            [chunk["text"] for chunk in output["chunks"]]).strip()
                    else:
                        transcript = ""
                else:
                    transcript = str(output).strip()

                patient_row = df_mmse[df_mmse["adressfname"]
                                      == audio_file.stem]
                segment_path = csv_segment_path + \
                    f"/{diagnosis}/{audio_file.stem}.csv"

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
                    transcript = " ".join([chunk["text"]
                                          for chunk in output["chunks"]]).strip()
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


def transcript_pipeline() -> None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Get files
    path_config = io.load_yaml("src/config/path.yaml")
    audio_train_files = io.get_files(
        path_config["train"]["AUDIO_TRAIN_PATH"], data_type="train")
    audio_test_files = io.get_files(
        path_config["test"]["AUDIO_TEST_PATH"], data_type="test")

    model_config = io.load_yaml("src/config/model.yaml")
    transcribe_audio_files(audio_train_files,
                           path_config["train"]["MMSE_DIAG_TRAIN_PATH"],
                           path_config["train"]["CSV_SEGMENT_TRAIN_PATH"],
                           data_type="train",
                           multipleGPU=model_config["whisper"]["MULTIPLE_GPU"],
                           model_name=model_config["whisper"]["MODEL_NAME"],
                           batch_size=model_config["whisper"]["BATCH_SIZE"],
                           device=DEVICE,
                           output_path=path_config["TRANSCRIPT_PATH"])
    transcribe_audio_files(audio_test_files,
                           path_config["test"]["MMSE_DIAG_TEST_PATH"],
                           path_config["test"]["CSV_SEGMENT_TEST_PATH"],
                           data_type="test",
                           multipleGPU=model_config["whisper"]["MULTIPLE_GPU"],
                           model_name=model_config["whisper"]["MODEL_NAME"],
                           batch_size=model_config["whisper"]["BATCH_SIZE"],
                           device=DEVICE,
                           output_path=path_config["TRANSCRIPT_PATH"])
    print("Transcript pipeline done")
