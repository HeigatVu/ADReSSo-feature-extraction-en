from src.traditionalApproach import featureSelection
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import pandas as pd


def specificity_score(y_true: np.ndarray,
                      y_pred: np.ndarray) -> float:
    """ Compute specificity
    """
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def sensitivity_score(y_true: np.ndarray,
                      y_pred: np.ndarray) -> float:
    """ Compute sensitivity
    """
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
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


def confusion_matrix_func(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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

# Merged feauture


def build_selector(strategy: str, test_case: list, threshold: float = 0.0):
    if strategy == "pca":
        return featureSelection.PCASelector(n_components=test_case)
    else:
        return featureSelection.HybridFeatureSelector(threshold=threshold, k=test_case)


def scale_and_select(X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_test: pd.DataFrame,
                     strategy: str,
                     test_case: list,
                     threshold: float = 0.0,
                     tag: str = "") -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, featureSelection]:
    """ Scale and select features
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    selector = build_selector(strategy, test_case, threshold)
    selector.fit_transform(X_train_scaled, y_train)

    if strategy == "pca":
        n_components = selector.n_components_
        cols_train = []
        for i in range(n_components):
            cols_train.append(f"{tag}_pc{i}")
        X_train_sel = pd.DataFrame(selector.transform(
            X_train_scaled), columns=cols_train, index=X_train.index)
        X_test_sel = pd.DataFrame(selector.transform(
            X_test_scaled), columns=cols_train, index=X_test.index)
    else:
        sel_cols = selector.selected_features_
        tagged = []
        for col in sel_cols:
            tagged.append(f"{tag}_{col}")
        X_train_sel = X_train_scaled.iloc[:, selector.selected_indices_].copy()
        X_test_sel = X_test_scaled.iloc[:, selector.selected_indices_].copy()
        X_train_sel.columns = tagged
        X_test_sel.columns = tagged

    return X_train_sel, X_test_sel, scaler, selector


def fused_feature(X_ling_df_train: pd.DataFrame,
                  X_acoustic_df_train: pd.DataFrame,
                  X_compare_df_train: pd.DataFrame,
                  y_train: pd.Series = None,
                  X_linguistic_df_test: pd.DataFrame = None,
                  X_acoustic_df_test: pd.DataFrame = None,
                  X_compare_df_test: pd.DataFrame = None,
                  strategy: str = "hybrid",
                  test_case: list = [],
                  threshold: float = 0.0) -> tuple[pd.DataFrame, pd.Series]:
    """ Fused linguistic with acoustic feature
    """
    if len(X_ling_df_train) == len(X_acoustic_df_train):
        # Linguisti feature
        X_ling, y_ling = X_ling_df_train.drop(
            columns=["label"]), X_ling_df_train["label"]
        X_ling_train_sel, X_ling_test_sel, _, _ = scale_and_select(X_ling_df_train, y_train, X_linguistic_df_test,
                                                                   strategy, test_case, threshold, "ling")

        # Acoustic feature
        X_ac, y_ac = X_acoustic_df_train.drop(
            columns=["label"]), X_acoustic_df_train["label"]
        X_ac_train_sel, X_ac_test_sel, _, _ = scale_and_select(X_acoustic_df_train, y_train, X_acoustic_df_test,
                                                               strategy, test_case, threshold, "acoustic")

        # Compare feature
        if (X_compare_df_train is not None) and \
            (len(X_compare_df_train) == len(X_ling_df_train)) and \
                (len(X_compare_df_train) == len(X_acoustic_df_train)):

            X_compare, y_compare = X_compare_df_train.drop(
                columns=["label"]), X_compare_df_train["label"]
            X_compare_train_sel, X_compare_test_sel, _, _ = scale_and_select(X_compare_df_train, y_train, X_compare_df_test,
                                                                             strategy, test_case, threshold, "compare")
            X_fused_train = pd.concat([X_ling_train_sel.reset_index(drop=True),
                                       X_ac_train_sel.reset_index(drop=True),
                                       X_compare_train_sel.reset_index(drop=True)], axis=1)
            X_fused_test = pd.concat([X_ling_test_sel.reset_index(drop=True),
                                      X_ac_test_sel.reset_index(drop=True),
                                      X_compare_test_sel.reset_index(drop=True)], axis=1)
        else:
            X_fused_train = pd.concat([X_ling_train_sel.reset_index(drop=True),
                                       X_ac_train_sel.reset_index(drop=True)], axis=1)
            X_fused_test = pd.concat([X_ling_test_sel.reset_index(drop=True),
                                      X_ac_test_sel.reset_index(drop=True)], axis=1)
    else:
        raise ValueError(
            "The number of rows in the linguistic and acoustic features is not the same")

    return X_fused_train, X_fused_test
