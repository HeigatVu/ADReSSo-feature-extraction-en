from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd



def feature_selection_ANOVA(df_train:pd.DataFrame, df_test:pd.DataFrame, alpha:float=0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select features using ANOVA
    """
    pass
    return df_train, df_test