import numpy as np
from scipy.stats import loguniform, randint, uniform

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import xgboost as xgb

def create_models(seed:int=42) -> dict:
    """ Create classifiers
    """

    lr = LogisticRegression(random_state=seed, max_iter=10000)
    svm = SVC(probability=True, random_state=seed)
    rf = RandomForestClassifier(criterion="gini", random_state=seed, n_jobs=1)
    mlp = MLPClassifier(random_state=seed, 
                        hidden_layer_sizes=(400,),
                        activation="logistic",
                        solver="sgd",
                        learning_rate="adaptive",
                        learning_rate_init=1e-3,
                        batch_size="auto",
                        max_iter=10000)
    xgb_model = xgb.XGBClassifier(random_state=seed,
                            eval_metric="logloss",
                            n_jobs=1)

    return {
        "lr": lr,
        "svm": svm,
        "rf": rf,
        "mlp": mlp,
        "xgb": xgb_model
    }


def create_hyperparameter_space(use_pipeline:bool=True) -> dict:
    """ Create hyperparameter space for classifiers
    """

    if use_pipeline:
        prefix = "clf__"
    else:
        prefix = ""

    lr_params = {
        prefix + "C": loguniform(1e-5, 1e2),
        prefix + "penalty": ["l1", "l2"],
        prefix + "solver": ["liblinear", "saga"],
    }

    svm_params = {
        prefix + "C": loguniform(1e-5, 1e2),
        prefix + "gamma": ["scale", "auto"] + list(np.geomspace(1e-6, 1.0, 10)),
        prefix + "kernel": ["rbf", "linear"],
    }

    rf_params = {
        prefix + "n_estimators": list(range(50, 501, 50)),
        prefix + "max_depth": [None] + list(range(5, 31, 5)),
        prefix + "min_samples_split": [2, 3, 4, 5],
        prefix + "min_samples_leaf": [1, 2, 3],


    }

    mlp_params = {
        prefix + "learning_rate_init": loguniform(1e-3, 1e-2),
        prefix + "batch_size": [16, 32, 64, 128, 166],
        prefix + "alpha": loguniform(1e-4, 1e-3)
    }

    xgb_params = {
        prefix + "learning_rate": uniform(1e-2, 0.49),
        prefix + "n_estimators": randint(50, 501),
        prefix + "max_depth": randint(1, 10),
        prefix + "subsample": uniform(1e-2, 0.99),
        prefix + "colsample_bytree": uniform(0.7, 0.3),
        prefix + "reg_alpha": uniform(0.0, 1e-3),
        prefix + "gamma": uniform(0, 0.5)
    }

    return {
        "lr": lr_params,
        "svm": svm_params,
        "rf": rf_params,
        "mlp": mlp_params,
        "xgb": xgb_params,
    }