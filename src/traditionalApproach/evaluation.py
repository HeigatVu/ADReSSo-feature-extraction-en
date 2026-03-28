from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, make_scorer

from src.traditionalApproach import tuning, modelsML

from sklearn import set_config
set_config(transform_output="pandas")

def evaluate_all_models(X_train:pd.DataFrame,
                        y_train:pd.Series,
                        feat_type:str="compare", 
                        strategy:str="hybrid", 
                        n_iter:int=20, 
                        cv:int=10) -> tuple[pd.DataFrame, dict]:
    """ Runs RandomizedSearchCV across all models for a given dataset and feature selection strategy.
    """
    # Load model
    models = modelsML.create_models()
    cls_params = tuning.tuning_hyperparameter_model()
    # Choose selector key
    if strategy == "pca":
        selector_key = "pca__n_components"
    else:
        selector_key = "feat_sel__k"
    
    best_estimators = {}
    pbar = tqdm(models.items(), desc=f"Tuning [{strategy.upper()}] | {feat_type}")
    for name, clf in pbar:
        pbar.set_description(f"[{strategy.upper()}] | {feat_type} {name.upper()}")

        pipeline, selector_grid = tuning.build_pipeline(clf, strategy)

        # Merged grid for testing
        merged_grid = {**cls_params[name], **selector_grid}

        scoring = {
            "sensitivity": make_scorer(recall_score),  # Sensitivity is the same as recall
            "specificity": make_scorer(specificity_score),
            "roc_auc": "roc_auc",
            "accuracy": make_scorer(accuracy_score)
        }


        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=merged_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            refit="accuracy",
        )

        random_search.fit(X_train, y_train)

        best_estimators[name.upper()] = random_search
        results = []

        best_params = random_search.best_params_
        best_selector_val = best_params.get(selector_key, "N/A")

        cls_best = dict()
        for k, v in best_params.items():
            if k != selector_key:
                cls_best[k] = v

        results.append({
            "Model": name.upper(),
            "Selector_Param": selector_key,
            "Best_Selector_Value": best_selector_val,
            "Best_CV_ROC_AUC": round(search.best_score_, 4),
            "Best_Params": str(cls_best),
        })

        tqdm.write(
            f"{name.upper():5s} | {selector_key}={best_selector_val} | CV ROC-AUC={search.best_score_:4f}"
        )

    df_results = pd.DataFrame(results).sort_values("Best_CV_ROC_AUC", ascending=False)
    print(df_results[
        ["Model", "Selector_Param", "Best_Selector_Value", "Best_CV_ROC_AUC"]
    ].to_markdown(index=False))

    return df_results, best_estimators


def evaluate_test_set(best_estimators: dict, 
                        X_test:pd.DataFrame,
                        y_test:pd.Series,
                        strategy:str="hybrid") -> pd.DataFrame:
    """ Evaluates fitted RandomizedSearchCV pipelines on a hold-out test set.
    """
    if strategy == "pca":
        selector_key = "pca__n_components"
    else:
        selector_key = "feat_sel__k"
    
    results = []
    
    for name, search_obj in best_estimators.items():
        # 1. Extract the best fitted pipeline
        best_pipeline = search_obj.best_estimator_
        
        # 2. Pipeline automatically scales and filters features, then predicts
        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        
        # 3. Calculate metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        
        # Calculate Sensitivity (True Positive Rate) and Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        if (tp + fn) > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 0
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0
        
        results.append({
            "Model": name.upper(),
            "Sensitivity": round(sensitivity, 4),
            "Specificity": round(specificity, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Accuracy": round(acc, 4),
            "Best_Selector_Value": search_obj.best_params_.get(selector_key, "N/A"),
        })
        
    df_results = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    return df_results