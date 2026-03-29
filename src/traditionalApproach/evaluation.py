from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import pandas as pd
from sklearn import set_config
set_config(transform_output="pandas")

from src.utils import helperFn
from src.traditionalApproach import tuning, modelsML

# Feature selection
def evaluate_selection_models(X_train:pd.DataFrame,
                        y_train:pd.Series,
                        strategy:str="hybrid",
                        threshold:float=0.0,
                        n_iter:int=20, 
                        cv:int=10) -> tuple[pd.DataFrame, dict]:
    """ Runs RandomizedSearchCV across all models for a given dataset and feature selection strategy.
    """
    model_config = io.load_yaml("src/config/model.yaml")
    seed = model_config["SEED"]

    # Load model
    models = modelsML.create_models()
    cls_params = tuning.tuning_hyperparameter_model()
    # Choose selector key
    if strategy == "pca":
        selector_key = "pca__n_components"
    else:
        selector_key = "feat_sel__k"
    
    scoring = helperFn.building_scoring()

    best_estimators = {}
    results = []

    for name, clf in models.items():

        pipeline, selector_grid = tuning.build_pipeline(clf, strategy, threshold)

        # Merged grid for testing
        merged_grid = {**cls_params[name], **selector_grid}

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=merged_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=seed,
            refit="accuracy",
        )

        random_search.fit(X_train, y_train)

        best_estimators[name.upper()] = random_search

        best_params = random_search.best_params_
        best_selector_val = best_params.get(selector_key, "N/A")
        cls_best = dict()
        for k, v in best_selector_val.items():
            if k != selector_key:
                cls_best[k] = v

        results.append({
            "Model": name.upper(),
            "Selector_Param": selector_key,
            "Best_Selector_Value": best_selector_val,
            "Best_CV_Accuracy": round(random_search.best_score_, 3),
            "Best_Params": str(cls_best),
        })

    df_results = pd.DataFrame(results).sort_values("Best_CV_accuracy", ascending=False)
    print(df_results[
        ["Model", "Selector_Param", "Best_Selector_Value", "Best_CV_accuracy"]
    ].to_markdown(index=False))

    return df_results, best_estimators


def evaluate_selection_test_set(best_estimators: dict, 
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
        best_pipeline = search_obj.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        metric_results = metrics.calculate_metrics(y_test, y_pred, y_prob)

        results.append({
            "Model": name.upper(),
            "Sensitivity": round(metric_results["sensitivity"], 4),
            "Specificity": round(metric_results["specificity"], 4),
            "ROC-AUC": round(metric_results["roc-auc"], 4),
            "Accuracy": round(metric_results["accuracy"], 4),
            "Best_Selector_Value": search_obj.best_params_.get(selector_key, "N/A"),
        })
        
    df_results = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    return df_results


# Raw feature selection
def evaluate_baseline_models(X_train:pd.DataFrame,
                                y_train:pd.Series,
                                n_iter:int=20,
                                cv:int=10) -> tuple[pd.DataFrame, dict]:

    """ Run RandomizedSearchCV with raw features
    """

    model_config = io.load_yaml("src/config/model.yaml")
    seed = model_config["SEED"]

    # Load model
    models = modelsML.create_models()
    cls_params = tuning.tuning_hyperparameter_model()
    scoring = helperFn.building_scoring()

    best_estimators = dict()
    results = []

    for name, clf in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=cls_params[name],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=seed,
            refit="accuracy",
        )

        random_search.fit(X_train, y_train)

        best_estimators[name.upper()] = random_search
        results.append({
            "Model": name.upper(),
            "Best_CV_Accuracy": round(random_search.best_score_, 3),
            "Best_Params": str(random_search.best_params_)
        })

    df_results = pd.DataFrame(results).sort_values("Best_CV_accuracy", ascending=False)
    print(df_results[
        ["Model", "Best_Params", "Best_CV_Accuracy"]
    ].to_markdown(index=False))

    return df_results, best_estimators


def evaluate_baseline_models_test_set(best_estimators:dict,
                                        X_test:pd.DataFrame,
                                        y_test:pd.Series,
                                        ) -> pd.DataFrame:
    """ Evaluate fitted baseline pipelines on the hold-out test set
    """
    results = []
    
    for name, search_obj in best_estimators.items():
        best_pipeline = search_obj.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        metric_results = metrics.calculate_metrics(y_test, y_pred, y_proba)
 
        results.append({
            "Model": name.upper(),
            "Sensitivity": round(metric_results["sensitivity"], 4),
            "Specificity": round(metric_results["specificity"], 4),
            "ROC-AUC": round(metric_results["roc-auc"], 4),
            "Accuracy": round(metric_results["accuracy"], 4),
        })
        
    df_results = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    return df_results


# Merged feature