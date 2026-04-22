from src.traditionalApproach import evaluation
from src.utils import io
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
# from sklearn import set_config
# set_config(transform_output="pandas")


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

    if feature_selection:
        pbar = tqdm(tests.items(), desc=f"Tuning")
        for strategy, feature_sets in tests.items():
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

                # Train and Evaluate
                df_train_results, fitted_models = evaluation.evaluate_selection_models(
                    X_train, y_train, strategy=strategy, correlation_threshold=correlation_threshold)
                test_metrics = evaluation.evaluate_selection_test_set(
                    fitted_models, X_test, y_test, strategy=strategy)

                print(
                    f"\nTest Results ({feat_type} | {strategy.upper()}):")
                print(test_metrics.to_markdown(index=False))

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
        df_summary = pd.DataFrame(summary_rows)

        output_dir = path_config["output_model"]['TRADITIONAL_MODEL_PATH']
        csv_path = os.path.join(
            output_dir, f"{output_csv_name}.csv")
        df_summary.to_csv(csv_path, index=False)
    else:
        pbar = tqdm(tests.items(), desc=f"Tuning")
        for strategy, feature_sets in tests.items():
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

                # Train and Evaluate
                df_train_results, fitted_models = evaluation.evaluate_baseline_models(
                    X_train, y_train)
                test_metrics = evaluation.evaluate_baseline_models_test_set(
                    fitted_models, X_test, y_test)

                print(
                    f"\nTest Results ({feat_type} | {strategy.upper()}):")
                print(test_metrics.to_markdown(index=False))

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

            df_summary = pd.DataFrame(summary_rows)
            # Save CSV
            output_dir = path_config["output_model"]['TRADITIONAL_MODEL_PATH']
            csv_path = os.path.join(
                output_dir, f"{output_csv_name}.csv")
            df_summary.to_csv(csv_path, index=False)

    print("Finish test")


def late_fusion_pipeline(acoustic_type: str,
                         linguistic_type: str,
                         strategy: str = "hybrid",
                         correlation_threshold: float = 0.0,
                         output_csv_name: str = "late_fusion_results") -> None:
    """ Late fusion pipeline: trains independent models for two sets and averages predictions.
    """
    from src.traditionalApproach import helperFn
    import numpy as np

    FEATURE_DISPLAY_NAMES = {
        "compare": "ComParE 2016",
        "egemaps": "eGeMAPS",
        "linguistic": "Linguistic Features",
        "praat": "Praat"
    }

    path_config = io.load_yaml("src/config/path.yaml")
    
    # 1. Load Acoustic Data
    df_csv_ac_tr = pd.read_csv(f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{acoustic_type}_train.csv")
    df_ac_tr = io.load_data(f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{acoustic_type}_train.pkl", df_csv=df_csv_ac_tr)
    X_train_ac, y_train = df_ac_tr.drop(columns=["label"]), df_ac_tr["label"]
    
    df_csv_ac_te = pd.read_csv(f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{acoustic_type}_test.csv")
    df_ac_te = io.load_data(f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{acoustic_type}_test.pkl", df_csv=df_csv_ac_te)
    X_test_ac, y_test = df_ac_te.drop(columns=["label"]), df_ac_te["label"]

    # 2. Load Linguistic Data
    df_csv_li_tr = pd.read_csv(f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{linguistic_type}_train.csv")
    df_li_tr = io.load_data(f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{linguistic_type}_train.pkl", df_csv=df_csv_li_tr)
    X_train_li = df_li_tr.drop(columns=["label"])
    
    df_csv_li_te = pd.read_csv(f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{linguistic_type}_test.csv")
    df_li_te = io.load_data(f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{linguistic_type}_test.pkl", df_csv=df_csv_li_te)
    X_test_li = df_li_te.drop(columns=["label"])

    # 3. Train and Evaluate Independent Models
    print(f"Training {acoustic_type} models...")
    df_ac_res, fitted_models_ac = evaluation.evaluate_selection_models(
        X_train_ac, y_train, strategy=strategy, correlation_threshold=correlation_threshold)

    print(f"Training {linguistic_type} models...")
    df_li_res, fitted_models_li = evaluation.evaluate_selection_models(
        X_train_li, y_train, strategy=strategy, correlation_threshold=correlation_threshold)

    # 4. Late Fusion (Average predictions)
    summary_rows = []
    for name in fitted_models_ac.keys():
        if name not in fitted_models_li:
            continue
            
        best_pipeline_ac = fitted_models_ac[name].best_estimator_
        best_pipeline_li = fitted_models_li[name].best_estimator_

        # Get probabilities for test set
        y_proba_ac = best_pipeline_ac.predict_proba(X_test_ac)[:, 1]
        y_proba_li = best_pipeline_li.predict_proba(X_test_li)[:, 1]

        # Late fusion (average)
        y_proba_fusion = (y_proba_ac + y_proba_li) / 2.0
        y_pred_fusion = (y_proba_fusion >= 0.5).astype(int)

        metric_results = helperFn.calculate_metrics(y_test.values, y_pred_fusion, y_proba_fusion)
        
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
        
    df_summary = pd.DataFrame(summary_rows).sort_values(by="Accuracy", ascending=False)
    
    print("\nTest Results (Late Fusion):")
    print(df_summary.to_markdown(index=False))

    output_dir = path_config["output_model"]['TRADITIONAL_MODEL_PATH']
    csv_path = os.path.join(output_dir, f"{output_csv_name}.csv")
    df_summary.to_csv(csv_path, index=False)
    
    print(f"\nLate Fusion Results saved to {csv_path}")
    
def early_fusion_pipeline(acoustic_type:str,
                          linguistic_type:str,
                          strategy:str="hybrid",
                          ):
