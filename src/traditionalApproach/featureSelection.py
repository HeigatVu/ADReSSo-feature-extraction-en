import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif

from mrmr import mrmr_classif

from src.utils import io
import joblib

# Initialize cache in a local directory
memory = joblib.Memory(".cache/mrmr", verbose=0)


@memory.cache
def _get_cached_mrmr(df, y, k_max=100, relevance='f', redundancy='c'):
    return mrmr_classif(X=df, y=y, K=k_max, relevance=relevance, redundancy=redundancy)


path_config = io.load_yaml("src/config/model.yaml")
seed = path_config["SEED"]


class PCASelector(BaseEstimator, TransformerMixin):
    """ PCA feature selection
    """

    def __init__(self, n_components: float = 0.95, random_state: int = seed):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        self._pca = PCA(n_components=self.n_components,
                        random_state=self.random_state)
        self._pca.fit(X)
        return self

    def transform(self, X):
        return self._pca.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class HybridFeatureSelector(BaseEstimator, TransformerMixin):
    """ Union of top-k features from ANOVA, Random Forest, and MRMR
    """

    def __init__(self, k: int = 10, random_state: int = seed, correlation_threshold: float = 0.0):
        self.k = k
        self.random_state = random_state
        self.correlation_threshold = correlation_threshold

    def fit(self, X, y=None):
        # Normalise input to DataFrame so mrmr_classif can use column names
        df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        df.columns = df.columns.astype(str)
        df = df.reset_index(drop=True)
        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)

        # 1. ANOVA
        self._anova = SelectKBest(f_classif, k=self.k).fit(df, y)
        anova_cols = set(df.columns[self._anova.get_support()])

        # 2. Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=1)
        self._rf_sel = SelectFromModel(
            rf, max_features=self.k, threshold=-np.inf).fit(df, y)
        rf_cols = set(df.columns[self._rf_sel.get_support()])

        # 3. MRMR (correlation-based redundancy)
        ranking_mrmr = _get_cached_mrmr(df, y, k_max=100)
        mrmr_cols = set(ranking_mrmr[:self.k])

        # Union of all methods
        merged = anova_cols | rf_cols | mrmr_cols

        if self.correlation_threshold > 0.0:
            # Adding to remove correrated features
            df_selected = df[sorted(merged)]
            corr_matrix = df_selected.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            redundant_features = set()
            for col in upper.columns:
                if any(upper[col] >= self.correlation_threshold):
                    redundant_features.add(col)
            merged = sorted(set(merged) - redundant_features)

        self.selected_features_ = sorted(merged)
        self.selected_indices_ = [df.columns.get_loc(
            c) for c in self.selected_features_]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_].values
        return X[:, self.selected_indices_]


class IntersectionFeatureSelector(BaseEstimator, TransformerMixin):
    """ Select features based on intersection of top-k candidates from ANOVA, Random Forest, and mRMR.
    """

    def __init__(self, k: int = 10, k_internal: int = 50, random_state: int = seed):
        self.k = k
        self.k_internal = k_internal
        self.random_state = random_state

    def fit(self, X, y=None):
        # Normalise input to DataFrame so mrmr_classif can use column names
        df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        df.columns = df.columns.astype(str)
        df = df.reset_index(drop=True)
        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)

        # Ensure k_internal doesn't exceed number of features
        n_features = df.shape[1]
        actual_k_internal = min(self.k_internal, n_features)

        # 1. ANOVA candidates
        anova = SelectKBest(f_classif, k=actual_k_internal).fit(df, y)
        anova_cols = df.columns[anova.get_support()].tolist()

        # 2. Random Forest candidates
        rf = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=1)
        rf_sel = SelectFromModel(
            rf, max_features=actual_k_internal, threshold=-np.inf).fit(df, y)
        rf_cols = df.columns[rf_sel.get_support()].tolist()

        # 3. MRMR candidates (ranking_mrmr is ordered by importance)
        ranking_mrmr = _get_cached_mrmr(df, y, k_max=max(100, self.k_internal))
        mrmr_cols = ranking_mrmr[:actual_k_internal]
        mrmr_full_ranking = {feat: i for i, feat in enumerate(ranking_mrmr)}

        # Voting logic: count how many methods selected each candidate
        votes = {}
        for col in set(anova_cols + rf_cols + mrmr_cols):
            count = 0
            if col in anova_cols:
                count += 1
            if col in rf_cols:
                count += 1
            if col in mrmr_cols:
                count += 1
            votes[col] = count

        # Filter to features with at least 2 votes
        candidates = [col for col, count in votes.items() if count >= 2]

        # Sort by: vote count (descending), then mRMR rank (ascending) as tiebreaker
        candidates.sort(
            key=lambda c: (-votes[c], mrmr_full_ranking.get(c, 999)))

        selected = candidates[:self.k]

        # Padding logic: if fewer than k features met the intersection criteria,
        # fill the remaining slots with the next best features from mRMR ranking
        if len(selected) < self.k:
            for col in ranking_mrmr:
                if col not in selected:
                    selected.append(col)
                if len(selected) == self.k:
                    break

        self.selected_features_ = selected
        self.selected_indices_ = [df.columns.get_loc(
            c) for c in self.selected_features_]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_indices_].values
        return X[:, self.selected_indices_]
