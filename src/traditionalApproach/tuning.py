from scipy.stats import loguniform, randint, uniform
import numpy as np
import src.traditionalApproach.featureSelection as featureSelection

def tuning_hyperparameter_model(use_pipeline:bool=True) -> dict:
    """ Create hyperparameter space for classifiers
    """

    if use_pipeline:
        prefix_model = "clf__"
    else:
        prefix_model = ""

    lr_params = {
        prefix_model + "C": loguniform(1e-5, 1e2),
        prefix_model + "penalty": ["l1", "l2"],
        prefix_model + "solver": ["liblinear", "saga"],
    }

    svm_params = {
        prefix_model + "C": loguniform(1e-5, 1e2),
        prefix_model + "gamma": ["scale", "auto"] + list(np.geomspace(1e-6, 1.0, 10)),
        prefix_model + "kernel": ["rbf", "linear"],
    }

    rf_params = {
        prefix_model + "n_estimators": list(range(50, 501, 50)),
        prefix_model + "max_depth": [None] + list(range(5, 31, 5)),
        prefix_model + "min_samples_split": [2, 3, 4, 5],
        prefix_model + "min_samples_leaf": [1, 2, 3],


    }

    mlp_params = {
        prefix_model + "learning_rate_init": loguniform(1e-3, 1e-2),
        prefix_model + "batch_size": [16, 32, 64, 128, 166],
        prefix_model + "alpha": loguniform(1e-4, 1e-3)
    }

    xgb_params = {
        prefix_model + "learning_rate": uniform(1e-2, 0.49),
        prefix_model + "n_estimators": randint(50, 501),
        prefix_model + "max_depth": randint(1, 10),
        prefix_model + "subsample": uniform(1e-2, 0.99),
        prefix_model + "colsample_bytree": uniform(0.7, 0.3),
        prefix_model + "reg_alpha": uniform(0.0, 1e-3),
        prefix_model + "gamma": uniform(0, 0.5)
    }

    return {
        "lr": lr_params,
        "svm": svm_params,
        "rf": rf_params,
        "mlp": mlp_params,
        "xgb": xgb_params,
    }

def pca_selector_hyperparameters() -> dict:
    """ Tuning PCA hyperparameter
    """

    pca_params = {
        "pca__n_components": [0.8, 0.85, 0.9, 0.95, 0.99]
    }

    return pca_params

def hybrid_selector_hyperparameter() -> dict:
    """ Tuning hybrid method
    """

    hybrid_params = {
        "feat_sel__k": list(range(5, 21, 5))
    }
    return hybrid_params

def build_pipeline(clf:str="lr", strategy:str="hybrid") -> tuple:
    """ Build pipeline
    """
    if strategy == "pca":
        selector_step = ("pca", featureSelection.PCASelector())
        selector_grid = pca_selector_grid()
    elif strategy == "hybrid":
        selector_step = ("feat_sel", featureSelection.HybridFeatureSelector())
        selector_grid = hybrid_selector_hyperparameter()
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        selector_step,
        ("clf", clf)
    ])

    return pipeline, selector_grid