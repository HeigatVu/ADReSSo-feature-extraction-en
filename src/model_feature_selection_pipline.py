import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(transform_output="pandas")
from src.models import modelsML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from src.models import featureSelection
from src.utils import io


def evaluate_all_models(X_train, y_train, strategy="hybrid", k=20, n_iter=20):
    """
    Runs RandomizedSearchCV across all models for a given dataset and feature selection strategy.
    """
    models = modelsML.create_models()
    params = modelsML.create_hyperparameter_space()
    best_estimators = {}

    pbar = tqdm(models.items(), desc=f"Strategy: {strategy.upper()}")
    for name, clf in pbar:
        pbar.set_description(f"Strategy: {strategy.upper()} | Training {name.upper()}")
        # Define the reduction step based on the test
        if strategy == "hybrid":
            reducer = ("feat_sel", featureSelection.HybridFeatureSelector(k=k))
        elif strategy == "pca":
            reducer = ("pca", featureSelection.PCASelector(n_components=0.95, random_state=42))

        # Build pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            reducer,
            ('clf', clf)
        ])

        # Execute Search
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params[name],
            n_iter=n_iter,
            scoring='accuracy', # Adjust to roc_auc if your classes are imbalanced
            cv=5,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_train, y_train)
        best_estimators[name] = search
        tqdm.write(f"{name.upper()} Best CV Accuracy: {search.best_score_:.4f}")

    return best_estimators

def evaluate_test_set(best_estimators: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluates fitted RandomizedSearchCV pipelines on a hold-out test set.
    """
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
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({
            "Model": name.upper(),
            "ROC-AUC": round(roc_auc, 4),
            "Accuracy": round(acc, 4),
            "Sensitivity": round(sensitivity, 4),
            "Specificity": round(specificity, 4)
        })
        
    df_results = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    return df_results

def model_pipeline(strategy:str="hybrid", feat_type:str="compare", k:int=10) -> None:
    path_config = io.load_yaml("src/config/path.yaml")
    # Load Train
    df_csv_tr = pd.read_csv(f"{path_config['OUTPUT_FEATURE_PATH']}/adresso_{feat_type}_train.csv")
    df_train = io.load_data(f"{path_config['PKL_PATH']}/adresso_{feat_type}_train.pkl", df_csv=df_csv_tr)
    X_train, y_train = df_train.drop(columns=["label"]), df_train["label"]
    
    # Load Test
    df_csv_te = pd.read_csv(f"{path_config['OUTPUT_FEATURE_PATH']}/adresso_{feat_type}_test.csv")
    df_test = io.load_data(f"{path_config['PKL_PATH']}/adresso_{feat_type}_test.pkl", df_csv=df_csv_te)
    X_test, y_test = df_test.drop(columns=["label"]), df_test["label"]

    # Train & Evaluate
    fitted_models = evaluate_all_models(X_train, y_train, strategy=strategy, k=k)
    test_metrics = evaluate_test_set(fitted_models, X_test, y_test)
    
    print(f"\nFinal Test Results ({feat_type}):")
    print(test_metrics.to_markdown(index=False))