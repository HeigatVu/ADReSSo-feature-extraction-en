from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import xgboost as xgb

def create_models() -> dict:
    """ Create classifiers
    """

    path_config = io.load_yaml("src/config/model.yaml")
    seed = path_config["SEED"]

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