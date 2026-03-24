from typing import List
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import pickle

def load_yaml(yaml_path: str) -> dict:
    """Load YAML file and set BASE_PATH to project root."""
    conf = OmegaConf.load(yaml_path)
    return conf


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

def csv_to_pkl(csv_path:str, 
                pkl_path:str, 
                label_col:str="diagnosis", 
                id_col:str="patient_id",
                mmse_col:str="mmse",
                lang_col:str="lang") -> None:
    """Convert CSV file to PKL file for processing
    """

    df = pd.read_csv(csv_path)
    df_copy = df.copy()
    
    feature_cols = []
    for col in df.columns:
        if col not in [id_col, label_col, mmse_col, lang_col]:
            feature_cols.append(col)
    
    # Handle missing value
    n_missing_value = df_copy[feature_cols].isnull().sum().sum()
    if n_missing_value > 0:
        df_copy[feature_cols] = df_copy[feature_cols].fillna(0)

    # Mapping label
    label_map = {
        "ad": 1,
        "cn": 0
    }

    df_copy["label"] = df_copy[label_col].map(label_map)
    df_copy["label"] = df_copy["label"].astype(int)

    out = pd.DataFrame({
        "pid": df_copy[id_col].astype(str).str.extract(r'(\d+)', expand=False).astype(int),
        "label": df_copy["label"],
        "mmse": df_copy[mmse_col],
        "data": list(df_copy[feature_cols].values.astype(np.float32))
    })

    Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(out, f)

def load_data(pkl_path:str, 
            feature_names:list=None,
            meta_data:bool=False, df_csv:str=None) -> pd.DataFrame:
    """Load data for feature selection and classification
    """
    df = pd.read_pickle(pkl_path)
    X_train = np.array(df["data"].tolist()) # Do not scale here

    data_expanded = pd.DataFrame(
        X_train,
        index=df.index
    )

    if df_csv is not None:
        feature_names = []
        for col in df_csv.columns:
            if col in ["patient_id", "diagnosis", "mmse", "lang"]:
                continue
            feature_names.append(col)
        data_expanded.columns = feature_names
    else:
        data_expanded.columns = [f'feature_{i}' for i in range(data_expanded.shape[1])]

    if meta_data:
        return pd.concat([df[['pid', 'mmse']], data_expanded, df['label']], axis=1)
    else:
        return pd.concat([data_expanded, df['label']], axis=1)