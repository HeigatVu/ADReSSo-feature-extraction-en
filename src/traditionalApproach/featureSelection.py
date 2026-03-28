import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif

from mrmr import mrmr_classif


class PCASelector(BaseEstimator, TransformerMixin):
    """ PCA feature selection
    """

    def __init__(self, n_components: float = 0.95, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self._pca.fit(X)
        return self

    def transform(self, X):
        return self._pca.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class HybridFeatureSelector(BaseEstimator, TransformerMixin):
    """ Union of top-k features from ANOVA, Random Forest, and MRMR
    """

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, X, y=None):
        # Normalise input to DataFrame so mrmr_classif can use column names
        df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        df.columns = df.columns.astype(str)

        # 1. ANOVA
        self._anova = SelectKBest(f_classif, k=self.k).fit(df, y)
        anova_cols = set(df.columns[self._anova.get_support()])

        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self._rf_sel = SelectFromModel(rf, max_features=self.k, threshold=-np.inf).fit(df, y)
        rf_cols = set(df.columns[self._rf_sel.get_support()])

        # 3. MRMR (difference & quotient)
        mrmr_diff = set(mrmr_classif(df, y, K=self.k))
        mrmr_quot = set(mrmr_classif(df, y, K=self.k, relevance="f", redundancy="c"))

        # Union of all methods
        merged = anova_cols | rf_cols | mrmr_diff | mrmr_quot
        self.selected_features_ = sorted(merged)
        self.selected_indices_  = [df.columns.get_loc(c) for c in self.selected_features_]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_].values
        return X[:, self.selected_indices_]