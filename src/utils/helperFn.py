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

def fused_feature(df_ling:pd.DataFrame,
                    df_acoustic:pd.DataFrame,
                    df_compare:pd.DataFrame=None) -> tuple[pd.DataFrame, pd.Series]:
    """ Fused linguistic with acoustic feature
    """
    if len(df_ling) == len(df_acoustic):
        # Linguisti feature
        X_ling, y_ling = df_ling.drop(columns=["label"]), df_ling["label"]
        
        # Acoustic feature
        X_ac, y_ac = df_acoustic.drop(columns=["label"]), df_acoustic["label"]

        if df_compare is not None and len(df_compare) == len(df_ling) and len(df_compare) == len(df_acoustic):
            X_compare, y_compare = df_compare.drop(columns=["label"]), df_compare["label"]
            X_fused = pd.concat([X_ling, X_ac, X_compare], axis=1)
        else:
            X_fused = pd.concat([X_ling, X_ac], axis=1)
    else:
        raise ValueError("The number of rows in the linguistic and acoustic features is not the same")
    
    return X_fused, y_ling