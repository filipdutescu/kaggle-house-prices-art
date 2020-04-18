from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, features :'list of strings', operation, as_dataframe :bool = False, feat_name :str = 'NewFeature'):
        self.features = features
        self.operation = operation
        self.as_dataframe = as_dataframe
        self.feat_name = feat_name

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        no_feat = len(self.features)
        prev_feat = self.features[0]
        for i in range(1, no_feat):
            new_feature = self.operation(X[prev_feat], X[self.features[i]])
            prev_feat = self.features[i]

        if self.as_dataframe:
            X[self.feat_name] = new_feature
            return X

        return np.c_[X, new_feature]


class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features, as_dataframe :bool = False):
        self.features = features
        self.as_dataframe = as_dataframe

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        if self.as_dataframe == True:
            return X.drop(columns=self.features)
        return np.c_[X.drop(columns=self.features)]


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features, as_dataframe :bool = False):
        self.features = features
        self.as_dataframe = as_dataframe

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        if self.as_dataframe == True:
            return X[self.features]
        return np.c_[X[self.features]]


class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, strategy :str = 'most_frequent', value :str = None):
        self.strategy = strategy
        self.value = value

    def fit(self, X, y=None):
        if self.strategy == 'most_frequent':
            self.fill = pd.Series([X[col].mode()[0] for col in X], index=X.columns)
        elif self.strategy == 'nan_to_none':
            self.fill = pd.Series(['None' for col in X], index=X.columns)
        elif self.strategy == 'custom_val' and self.value is not None:
            self.fill = pd.Series([self.value for col in X], index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


