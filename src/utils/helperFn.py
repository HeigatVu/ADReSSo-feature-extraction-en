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

def collect_cv_row(name:str, 
                    random_search:RandomizedSearchCV, 
                    selector_key:str) -> dict:
    """ Extract result row from a fitted Randomized Search CV
    """

    best_params = random_search.best_params_
    best_selector_val = best_params.get(selector_key, "N/A")
    cls_best = dict()
    for k, v in best_selector_val.items():
        if k != selector_key:
            cls_best[k] = v

    return {
        "Model": name.upper(),
        "Selector_Param": selector_key,
        "Best_Selector_Value": best_selector_val,
        "Best_CV_Accuracy": round(random_search.best_score_, 3),
        "Best_Params": str(cls_best),
    }

def collect_test_row(name:str,
                        search_obj:RandomizedSearchCV,
                        X_test:pd.DataFrame, y_test:pd.DataFrame,
                        selector_key:str) -> dict:
    """ 
    """
    best_pipeline = search_obj.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    metric_results = metrics.calculate_metrics(y_test, y_pred, y_prob)

    return {
        "Model": name.upper(),
        "Sensitivity": round(metric_results["sensitivity"], 4),
        "Specificity": round(metric_results["specificity"], 4),
        "ROC-AUC": round(metric_results["roc-auc"], 4),
        "Accuracy": round(metric_results["accuracy"], 4),
        "Best_Selector_Value": search_obj.best_params_.get(selector_key, "N/A"),
    }
