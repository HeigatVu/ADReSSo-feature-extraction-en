from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
import pandas as pd


def specificity_score(y_true:np.ndarray, 
                        y_pred:np.ndarray) -> float:
    """ Compute specificity
    """
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)

def sensitivity_score(y_true:np.ndarray, 
                        y_pred:np.ndarray) -> float:
    """ Compute sensitivity
    """
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def calculate_metrics(y_true:np.ndarray, y_pred:np.ndarray, y_proba:np.ndarray=None) -> dict:
    """ Compute metrics
    """
    acc = accuracy_score(y_true, y_pred)
    sens = sensitivity_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    if y_proba is not None:
        rocauc = roc_auc_score(y_true, y_proba)
    else:
        rocauc = roc_auc_score(y_true, y_pred)

    return {
        "sensitivity": sens,
        "specificity": spec,
        "roc-auc": rocauc,
        "accuracy": acc,
    }

def confusion_matrix_func(y_true:np.ndarray, y_pred:np.ndarray) -> dict:
    """ Compute confusion matrix
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

def building_scoring() -> dict:
    """ Extract score after CV evaluation
    """

    return {
        "sensitivity": make_scorer(sensitivity_score),
        "specificity": make_scorer(specificity_score),
        "roc_auc": "roc_auc",
        "accuracy": make_scorer(accuracy_score)
    }

def load_early_fusion_split(path_config:dict,
                            acoustic_type:str,
                            data_type:str,
                            add_compare:bool=False,) -> tuple[pd.DataFrame, pd.Series]:
    """ Fused linguistic with acoustic feature
    """

    pass