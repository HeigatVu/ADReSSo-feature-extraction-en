from src.traditionalApproach import evaluation
from src.utils import io
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
from src.traditionalApproach import helperFn
import numpy as np
from src.traditionalApproach.featureSelection import PCASelector, HybridFeatureSelector


def model_pipeline_one_feature(tests: dict,
                               feature_selection: bool = False,
                               correlation_threshold: float = 0.0,
                               output_csv_name: str = "results") -> None:

    FEATURE_DISPLAY_NAMES = {
        "compare": "ComParE 2016",
        "egemaps": "eGeMAPS",
        "linguistic": "Linguistic Features",
        "praat": "Praat"
    }

    path_config = io.load_yaml("src/config/path.yaml")
    # Save result
    save_result = f"{path_config["output_model"]['TRADITIONAL_MODEL_PATH']}/adresso.csv"
    Path(save_result).parent.mkdir(parents=True, exist_ok=True)

    # Summary row
    summary_rows = []

    pbar = tqdm(tests.items(), desc=f"Tuning")
    for strategy, feature_sets in pbar:
        pbar.set_description(f"Tuning {strategy.upper()}")
        pbar_feat = tqdm(feature_sets, desc=f"Feature Type")
        for feat_type in pbar_feat:
            pbar_feat.set_description(
                f"Feature Type {feat_type.upper()}")
            # Train
            df_csv_tr = pd.read_csv(
                f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{feat_type}_train.csv")
            df_train = io.load_data(
                f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{feat_type}_train.pkl", df_csv=df_csv_tr)
            X_train, y_train = df_train.drop(
                columns=["label"]), df_train["label"]

            # Load Test
            df_csv_test = pd.read_csv(
                f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{feat_type}_test.csv")
            df_test = io.load_data(
                f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{feat_type}_test.pkl", df_csv=df_csv_test)
            X_test, y_test = df_test.drop(
                columns=["label"]), df_test["label"]

            if feature_selection:
                # Train and Evaluate
                df_train_results, fitted_models = evaluation.evaluate_selection_models(
                    X_train, y_train, strategy=strategy, correlation_threshold=correlation_threshold)
                test_metrics = evaluation.evaluate_selection_test_set(
                    fitted_models, X_test, y_test, strategy=strategy)
            else:
                # Train and Evaluate
                df_train_results, fitted_models = evaluation.evaluate_baseline_models(
                    X_train, y_train)
                test_metrics = evaluation.evaluate_baseline_models_test_set(
                    fitted_models, X_test, y_test)
            best_row = test_metrics.iloc[0]
            summary_rows.append({
                "Feature": FEATURE_DISPLAY_NAMES.get(feat_type, feat_type),
                "Strategy": strategy.upper(),
                "Best_Model": best_row["Model"],
                "Sensitivity": best_row["Sensitivity"],
                "Specificity": best_row["Specificity"],
                "ROC-AUC": best_row["ROC-AUC"],
                "Accuracy": best_row["Accuracy"],
            })
            tqdm.write(
                f"Test Results ({feat_type} | {strategy.upper()}\n)"
                f"{test_metrics.to_markdown(index=False)}"
            )
    # Save CSV
    df_summary = pd.DataFrame(summary_rows).sort_values(
        by="Accuracy", ascending=False)

    output_dir = path_config["output_model"]['TRADITIONAL_MODEL_PATH']
    csv_path = os.path.join(
        output_dir, f"{output_csv_name}.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nFinish {feat_type} results saved to {csv_path}")


def late_fusion_pipeline(acoustic_type: str,
                         linguistic_type: str,
                         strategy: str = "hybrid",
                         correlation_threshold: float = 0.0,
                         output_csv_name: str = "late_fusion_results") -> None:
    """ Late fusion pipeline: trains independent models for two sets and averages predictions.
    """

    FEATURE_DISPLAY_NAMES = {
        "compare": "ComParE 2016",
        "egemaps": "eGeMAPS",
        "linguistic": "Linguistic Features",
        "praat": "Praat"
    }

    path_config = io.load_yaml("src/config/path.yaml")

    # 1. Load Acoustic Data
    df_csv_acoustic_train = pd.read_csv(
        f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{acoustic_type}_train.csv")
    df_acoustic_train = io.load_data(
        f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{acoustic_type}_train.pkl", df_csv=df_csv_acoustic_train)
    X_train_acoustic, y_train = df_acoustic_train.drop(
        columns=["label"]), df_acoustic_train["label"]

    df_csv_acoustic_test = pd.read_csv(
        f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{acoustic_type}_test.csv")
    df_acoustic_test = io.load_data(
        f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{acoustic_type}_test.pkl", df_csv=df_csv_acoustic_test)
    X_test_acoustic, y_test = df_acoustic_test.drop(
        columns=["label"]), df_acoustic_test["label"]

    # 2. Load Linguistic Data
    df_csv_ling_train = pd.read_csv(
        f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{linguistic_type}_train.csv")
    df_ling_train = io.load_data(
        f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{linguistic_type}_train.pkl", df_csv=df_csv_ling_train)
    X_train_ling = df_ling_train.drop(columns=["label"])

    df_csv_ling_test = pd.read_csv(
        f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{linguistic_type}_test.csv")
    df_ling_test = io.load_data(
        f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{linguistic_type}_test.pkl", df_csv=df_csv_ling_test)
    X_test_ling = df_ling_test.drop(columns=["label"])

    # 3. Train and Evaluate Independent Models
    df_acoustic_ressult, fitted_models_acoustic = evaluation.evaluate_selection_models(
        X_train_acoustic, y_train, strategy=strategy, correlation_threshold=correlation_threshold)

    df_ling_result, fitted_models_ling = evaluation.evaluate_selection_models(
        X_train_ling, y_train, strategy=strategy, correlation_threshold=correlation_threshold)

    # 4. Late Fusion (Average predictions)
    summary_rows = []
    for name in fitted_models_acoustic.keys():
        if name not in fitted_models_ling:
            continue

        best_pipeline_acoustic = fitted_models_acoustic[name].best_estimator_
        best_pipeline_ling = fitted_models_ling[name].best_estimator_

        # Get probabilities for test set
        y_proba_acoustic = best_pipeline_acoustic.predict_proba(X_test_acoustic)[
            :, 1]
        y_proba_ling = best_pipeline_ling.predict_proba(X_test_ling)[:, 1]

        # Late fusion (average)
        y_proba_fusion = (y_proba_acoustic + y_proba_ling) / 2.0
        y_pred_fusion = (y_proba_fusion >= 0.5).astype(int)

        metric_results = helperFn.calculate_metrics(
            y_test.values, y_pred_fusion, y_proba_fusion)

        summary_rows.append({
            "Feature": f"{FEATURE_DISPLAY_NAMES.get(acoustic_type, acoustic_type)} + {FEATURE_DISPLAY_NAMES.get(linguistic_type, linguistic_type)}",
            "Strategy": strategy.upper(),
            "Model": name,
            "Sensitivity": round(metric_results["sensitivity"], 4),
            "Specificity": round(metric_results["specificity"], 4),
            "ROC-AUC": round(metric_results["roc-auc"], 4),
            "Accuracy": round(metric_results["accuracy"], 4),
            "Fusion_Strategy": "Average"
        })

    df_summary = pd.DataFrame(summary_rows)sort_values(
        by="Accuracy", ascending=False)
    # Save CSV
    output_dir = path_config["output_model"]["TRADITIONAL_MODEL_PATH"]
    csv_path = os.path.join(
        output_dir, f"{output_csv_name}.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nEarly fusion Results saved to {csv_path}")
    output_dir = path_config["output_model"]["TRADITIONAL_MODEL_PATH"]
    csv_path = os.path.join(output_dir, f"{output_csv_name}.csv")
    df_summary.to_csv(csv_path, index=False)


def early_fusion_pipeline(acoustic_list: list,
                          linguistic_type: str,
                          strategy: str = "hybrid",
                          k: int = 10,
                          correlation_threshold: float = 0.0,
                          output_csv_name: str = "results_merged_feature_model") -> None:
    """ Early fusion pipeline: selectes top-k from two linguistic and acoustic features -> concatenates
    """

    FEATURE_DISPLAY_NAMES = {
        "compare": "ComParE 2016",
        "egemaps": "eGeMAPS",
        "linguistic": "Linguistic Features",
        "praat": "Praat"
    }
    summary_rows = []

    path_config = io.load_yaml("src/config/path.yaml")
    # Load Linguistic Data
    df_csv_ling_train = pd.read_csv(
        f"{path_config["OUTPUT_TRADITIONAL_FEATURE_PATH"]}/adresso_{linguistic_type}_train.csv")
    df_ling_train = io.load_data(
        f"{path_config["PKL_TRADITIONAL_PATH"]}/adresso_{linguistic_type}_train.pkl", df_csv=df_csv_ling_train)
    X_train_ling = df_ling_train.drop(columns=["label"])

    df_csv_ling_test = pd.read_csv(
        f"{path_config["OUTPUT_TRADITIONAL_FEATURE_PATH"]}/adresso_{linguistic_type}_test.csv")
    df_ling_test = io.load_data(
        f"{path_config["PKL_TRADITIONAL_PATH"]}/adresso_{linguistic_type}_test.pkl", df_csv=df_csv_ling_test)
    X_test_ling = df_ling_test.drop(columns=["label"])

    pbar_feat = tqdm(acoustic_list, desc=f"Feature Acoustic Type")
    for acoustic_type in pbar_feat:
        pbar_feat.set_description(
            f"Feature Type {acoustic_type.upper()}")
        # Load Acoustic Data
        df_csv_acoustic_train = pd.read_csv(
            f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{acoustic_type}_train.csv")
        df_acoustic_train = io.load_data(
            f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{acoustic_type}_train.pkl", df_csv=df_csv_acoustic_train)
        X_train_acoustic, y_train = df_acoustic_train.drop(
            columns=["label"]), df_acoustic_train["label"]

        df_csv_acoustic_test = pd.read_csv(
            f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{acoustic_type}_test.csv")
        df_acoustic_test = io.load_data(
            f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{acoustic_type}_test.pkl", df_csv=df_csv_acoustic_test)
        X_test_acoustic, y_test = df_acoustic_test.drop(
            columns=["label"]), df_acoustic_test["label"]

        X_train_combined = pd.concat([X_train_acoustic, X_train_ling], axis=1)
        X_test_combined = pd.concat([X_test_acoustic, X_test_ling], axis=1)

        X_train_combined.columns = [
            f"feat_{i}" for i in range(X_train_combined.shape[1])]
        X_test_combined.columns = X_train_combined.columns
        df_train_results, fitted_models = evaluation.evaluate_selection_models(
            X_train_combined, y_train, strategy=strategy, correlation_threshold=correlation_threshold)
        test_metrics = evaluation.evaluate_selection_test_set(
            fitted_models, X_test_combined, y_test, strategy=strategy)
        # Save summary
        best_row = test_metrics.iloc[0]
        summary_rows.append({
            "Feature": f"{FEATURE_DISPLAY_NAMES.get(acoustic_type, acoustic_type)} + "
            f"{FEATURE_DISPLAY_NAMES.get(linguistic_type, linguistic_type)}",
            "Strategy": strategy.upper(),
            "Best_Model": best_row["Model"],
            "Sensitivity": best_row["Sensitivity"],
            "Specificity": best_row["Specificity"],
            "ROC-AUC": best_row["ROC-AUC"],
            "Accuracy": best_row["Accuracy"],
        })
        tqdm.write(
            f"Test Results ({acoustic_type} {linguistic_type} | {strategy.upper()}\n)"
            f"{test_metrics.to_markdown(index=False)}"
        )

    df_summary = pd.DataFrame(summary_rows).sort_values(
        by="Accuracy", ascending=False)
    # Save CSV
    output_dir = path_config["output_model"]["TRADITIONAL_MODEL_PATH"]
    csv_path = os.path.join(
        output_dir, f"{output_csv_name}.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nEarly fusion Results saved to {csv_path}")
