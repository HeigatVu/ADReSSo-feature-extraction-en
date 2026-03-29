import pandas as pd
from sklearn import set_config
set_config(transform_output="pandas")
from pathlib import Path
import os
from tqdm import tqdm

from src.utils import io
from src.traditionalApproach import evaluation

def model_pipeline(tests:dict, 
                    early_fusion:bool=False, 
                    feature_selection:bool=False,
                    threshold:float=0.0) -> None:

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

    if early_fusion:
        
        if feature_selection:
            pass
        else:
            pass

    else:
        if feature_selection:
            pbar = tqdm(tests.items(), desc=f"Tuning")
            for strategy, feature_sets in tests.items():
                pbar.set_description(f"Tuning {strategy.upper()}")
                pbar_feat = tqdm(feature_sets, desc=f"Feature Type")
                for feat_type in pbar_feat:
                    pbar_feat.set_description(f"Feature Type {feat_type.upper()}")
                    # Train
                    df_csv_tr = pd.read_csv(f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{feat_type}_train.csv")
                    df_train = io.load_data(f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{feat_type}_train.pkl", df_csv=df_csv_tr)
                    X_train, y_train = df_train.drop(columns=["label"]), df_train["label"]
                
                    # Load Test
                    df_csv_test = pd.read_csv(f"{path_config['OUTPUT_TRADITIONAL_FEATURE_PATH']}/adresso_{feat_type}_test.csv")
                    df_test = io.load_data(f"{path_config['PKL_TRADITIONAL_PATH']}/adresso_{feat_type}_test.pkl", df_csv=df_csv_test)
                    X_test, y_test = df_test.drop(columns=["label"]), df_test["label"]

                    # Train and Evaluate
                    df_train_results, fitted_models = evaluation.evaluate_selection_models(X_train, y_train, strategy=strategy, feat_type=feat_type, threshold=threshold)
                    test_metrics = evaluation.evaluate_selection_test_set(fitted_models, X_test, y_test, strategy=strategy, threshold=threshold)

                    print(f"\nTest Results ({feat_type} | {strategy.upper()}):")
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
        else:
            pass

    # Save CSV
    df_summary = pd.DataFrame(summary_rows)

    output_dir = path_config["output_model"]['TRADITIONAL_MODEL_PATH']
    csv_path = os.path.join(output_dir, "results_selection_traditional_model.csv")
    df_summary.to_csv(csv_path, index=False)

    print("Finish test")