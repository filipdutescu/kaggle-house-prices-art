from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, features :'list of strings', operation :lambda, as_dataframe = False, feat_name = 'NewFeature'):
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
            new_feature = self.operation(prev_feat, X[self.features[i]])
            prev_feat = self.features[i]

        if self.as_dataframe:
            X[self.feat_name] = new_feature
            return X

        return np.c_[X, new_feature]


