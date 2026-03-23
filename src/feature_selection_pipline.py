from src.models import featureSelection

from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

def get_annova_ranking(df:pd.DataFrame, target_name:str) -> pd.Series:
    """ Get feature ranking using ANOVA
    """
    features = featureSelection.get_features(df, target_name)
    f_scores = f_classif(df[features], df[target_name])[0]
    f_series = pd.Series(f_scores, index=features)
    sorted_series = f_series.sort_values(ascending=False)
    return sorted_series.index.tolist()

def get_mrmr_ranking(df:pd.DataFrame, target_name:str, method="difference") -> pd.Series:
    """ Get feature ranking using MRMR with difference method
    """
    result = featureSelection.rank_features(df, target_name, method=method)
    return result
    

def get_random_forest_ranking(df:pd.DataFrame, target_name:str) -> pd.Series:
    """ Get feature ranking using Random Forest
    """
    features = featureSelection.get_features(df, target_name)
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(df[features], df[target_name])
    return pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).index

def compare_ranking_methods(df:pd.DataFrame, 
                            target_name:str, k:int = 10, 
                            save_csv:bool=True, save_path:str=None, name:str=None) -> dict[str, list[str]]:
    """ Compare feature ranking methods
    """
    dict_method = {
        "annova": get_annova_ranking(df, target_name)[:k],
        "mrmr_difference_based": get_mrmr_ranking(df, target_name, method="difference")[:k],
        "mrmr_quotient_based": get_mrmr_ranking(df, target_name, method="quotient")[:k],
        "random_forest": get_random_forest_ranking(df, target_name)[:k]
    }

    merged_important_feature = []
    for features in dict_method.values():
        merged_important_feature.extend(features)
    
    merged_important_feature = list(set(merged_important_feature))

    if save_csv:
        df_method = pd.DataFrame(merged_important_feature)
        df_method.to_csv(save_path + "/" + name + ".csv ", index=False)

    return merged_important_feature