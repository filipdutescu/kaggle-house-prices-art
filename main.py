import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as tick

from transformers import FeatureCreator, FeatureSelector, FeatureDropper 

from sklearn.pipeline import Pipeline, FeatureUnion 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer as Imputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV


# Plot correlations
def corr_plot(data :pd.DataFrame, feature :str, threshold=0.5, plot_type :str = 'scatter', y_lower_scale=True, same_fig=True, fig_size=(3, 4)):
    fig = plt.figure()
    corr_matrix = data.corr()
    i = 1
    for feat in corr_matrix.columns:
        if abs(corr_matrix[feat][feature]) > threshold and feat != feature:
            if same_fig == True:
                ax = fig.add_subplot(fig_size[0], fig_size[1], i)
                if plot_type == 'scatter':
                    ax.scatter(x=feat, y=feature, data=data)
                elif plot_type == 'hist':
                    ax.hist(x=feat, data=data)
                ax.set_xlabel(feat)
                if y_lower_scale == True:
                    ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.e'))
                plt.yticks(rotation=45)
                i = i + 1
            else:
                if plot_type == 'scatter':
                    plt.scatter(x=feat, y=feature, data=data)
                elif plot_type == 'hist':
                    plt.hist(x=feat, data=data)
                plt.xlabel(feat)
                plt.show()

    if same_fig == True:
        fig.tight_layout()
        plt.show()


# Load data from the csv file, droping columns if provided
def load_data(filename:str, columns:'list of strings' = None):
    result = pd.read_csv(filename)
    if columns is not None and len(columns) > 1:
        return result.drop(columns=columns)
    return result


# Print a brief, quick analysis of a dataframe to gain insight
def quick_analysis(data_frame:pd.DataFrame):
    print('\nAnalysis of dataframe:')
    print(data_frame.head())
    print(data_frame.info())
    print(data_frame.describe())


# Process data creating new features and encoding categorical features, returning resulting array
def process_data_pipeline(raw_data :pd.DataFrame, num_feat:'list of numbers', categ_feat :'list of strings' = None, categ_feat_vals:'list of strings' = None):
    num_pipeline = Pipeline([
#            ('drop_non_num', FeatureDropper(cat_cols, as_dataframe=True)),
            ('feat_sel', FeatureSelector(num_feat, True)),
#            ('Grade', FeatureCreator(['OverallCond', 'OverallQual'], lambda x, y: x / y, as_dataframe=True, feat_name='Grade')),
#            ('Age', FeatureCreator(['YrSold', 'YearBuilt'], lambda x,y: x - y, as_dataframe=True, feat_name='Age')),
#            ('RemodAge', FeatureCreator(['YrSold', 'YearRemodAdd'], lambda x,y: x - y, as_dataframe=True, feat_name='RemodAge')),
#            ('TotalSF', FeatureCreator(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], lambda x,y: x + y, as_dataframe=True, feat_name='TotalSF')),
            ('imputer_mean', Imputer(strategy='mean')),
            ('std_scaler', StandardScaler())
        ]) 
    if categ_feat is None:
        return num_pipeline.fit_transform(raw_data)

    categ_cols = [raw_data[col].unique() for col in categ_feat] if categ_feat_vals is None else categ_feat_vals

    cat_pipeline = Pipeline([
#            ('drop_non_cat', FeatureDropper(num_cols, as_dataframe=True)),
            ('feat_sel', FeatureSelector(categ_feat, True)),
            ('imputer_most_frequent', Imputer(missing_values=np.nan, strategy='most_frequent')),
            ('encode', OneHotEncoder(sparse=False) if categ_cols is None else OneHotEncoder(categories=categ_cols, sparse=False)),
        ])
    feat_union = FeatureUnion(transformer_list=[
            ('num_features', num_pipeline),
            ('cat_features', cat_pipeline),
        ])

    return feat_union.fit_transform(raw_data)


# Display cross validation score statistics
def print_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Std. deviation: ', scores.std(), '\n')


# Test regression model using the cross validation method
def cross_eval_model(model, data, labels, scoring :str = 'neg_mean_squared_error', no_cv :int = 10, neg_scoring :bool = True, display_scores :bool =True):
    scores = cross_val_score(model, data, labels,
            scoring=scoring,
            cv=no_cv)
    final_scores = np.sqrt(-scores) if neg_scoring is True else scores
    print_scores(final_scores)


def main():
#    no_data_cols = [ 'Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature' ]
#    raw_data = load_data('train.csv', no_data_cols)
    raw_data = load_data('train.csv')
#    raw_data.dropna(inplace=True)
    #quick_analysis(raw_data)

    #plt.hist(raw_data['SalePrice'])
    #plt.show()

#    non_numeric_cols = raw_data.loc[:, raw_data.dtypes == object]
#    quick_analysis(non_numeric_cols)

    #for col in non_numeric_cols.columns:
    #    print(non_numeric_cols[col].value_counts())
    #    input()

    #raw_data.drop(columns=[ 'YearBuilt', 'YearRemodAdd' ], inplace=True)
    #raw_data['PercUnfBsmt'] = raw_data['BsmtUnfSF'] / raw_data['TotalBsmtSF'] 

#    corr_matrix = raw_data.corr()
#    sale_correl = corr_matrix['SalePrice'].sort_values(ascending=False)
#    print(sale_correl)

#    sale_correl_cols = corr_matrix.where(abs(sale_correl) > 0.5).where(abs(sale_correl) < 1).columns
#    low_corr_cols = list(corr_matrix.where(abs(sale_correl) <= 0.5).columns)
#    low_corr_cols = list(set(low_corr_cols) & set(cat_data.columns))
#    print(low_corr_cols)

    train_labels = raw_data['SalePrice']
#    raw_data.drop(columns=no_use_cols, inplace=True)
#    quick_analysis(raw_data)
#    print(raw_data.info())

    num_cols = ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'YrSold', 'MoSold'] 
    cat_cols = [
#            'MSZoning', 
#            'Street', 
#            'Utilities', 
            'Neighborhood', 
            'ExterQual', 
            'ExterCond', 
            'BsmtQual', 
            'BsmtCond', 
#            'Heating', 
            'CentralAir', 
            'PavedDrive', 
            'SaleType', 
#            'SaleCondition'
        ]
    cat_cols_categs = [raw_data[col].unique() for col in cat_cols]
#    for col in cat_cols:
#        cat_cols_categs.append(raw_data[col].unique())
#    print(cat_cols_categs)

#    corr_plot(raw_data, 'SalePrice', fig_size=(4, 4))
#    corr_plot(raw_data, 'SalePrice', y_lower_scale=False, same_fig=False)
#    plt.hist(x=raw_data['SalePrice'])
#    plt.show()
#    corr_plot(raw_data, 'SalePrice', plot_type='hist', y_lower_scale=False, same_fig=False)

#    print(cat_cols)
#    print(num_data.info())
    train_feat = process_data_pipeline(raw_data, num_cols, cat_cols)
#    print(train_feat.info())
#    print(train_feat.describe())
#    print(train_feat.shape)

    linear_reg = LinearRegression()
    linear_reg.fit(train_feat, train_labels)

#    fit_data_test = raw_data.iloc[:5]
#    fit_label_test = train_labels.iloc[:5]
#    fit_data_test = process_data_pipeline(fit_data_test, num_cols, cat_cols, cat_cols_categs)
#    lr_pred = linear_reg.predict(fit_data_test)

#    print('\nPredictions:\t', list(lr_pred))
#    print('\nActual:\t', list(fit_label_test))

#    train_num_data = num_pipeline.fit_transform(fit_data_test)
#    train_cat_data = cat_pipeline.fit_transform(fit_data_test)
#    num_data = num_pipeline.fit_transform(raw_data)
#    cat_data = cat_pipeline.fit_transform(raw_data)

#    print('Num data shapes: ', num_data.shape, train_num_data.shape)
#    print('Cat data shapes: ', cat_data.shape, train_cat_data.shape)

    print(raw_data['SalePrice'].describe(), '\n')

#    lr_pred = linear_reg.predict(train_feat)
#    lr_err = mean_squared_error(train_labels, lr_pred)
#    lr_rmse = np.sqrt(lr_err)
#    print('Linear regression error: %d\n' % lr_rmse)

#    dec_tree = DecisionTreeRegressor()
#    dec_tree.fit(train_feat, train_labels)
#    dec_tree_pred = dec_tree.predict(train_feat)
#    dt_err = mean_squared_error(train_labels, dec_tree_pred)
#    dt_rmse = np.sqrt(dt_err)
#    print('Decision tree error: %d\n' % dt_rmse)

#    linear_reg = LinearRegression()
#    print('Linear regression:')
#    cross_eval_model(linear_reg, train_feat, train_labels)

#    dec_tree = DecisionTreeRegressor()
#    print('Decision tree scores:')
#    cross_eval_model(dec_tree, train_feat, train_labels)

#    rand_for = RandomForestRegressor()
#    print('Random forest:')
#    cross_eval_model(rand_for, train_feat, train_labels)
    
#    xgbst = XGBRegressor()
#    print('XGBoost regressor:')
#    cross_eval_model(xgbst, train_feat, train_labels)

    param_rand = [
            {'n_estimators': [3, 10, 30, 45], 'max_features': [10, 12, 16, 18, 20, 24]},
            {'bootstrap': [False], 'n_estimators': [3, 10, 30, 45], 'max_features': [6, 10, 12, 18]},
        ]
    
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_rand, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(train_feat, train_labels)
    print('Random forest best hyperparameters:')
    print(grid_search.best_params_, grid_search.best_score_)

    cv_results = grid_search.cv_results_
    for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
        print(np.sqrt(-mean_score), params)

    feat_imp = grid_search.best_estimator_.feature_importances_
    other_feat = ['Grade', 'RemodAge', 'TotalSF']
    all_features = num_cols
    for cat_values in cat_cols_categs:
        all_features.extend(cat_values)
    all_features.extend(other_feat)
    for feat in sorted(zip(feat_imp, all_features), reverse=True):
        print(feat)

    
if __name__ == '__main__':
    main()


