import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as tick

from transformers import FeatureCreator, FeatureSelector, FeatureDropper, CategoricalImputer

from sklearn.pipeline import Pipeline, FeatureUnion 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer as Imputer

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

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
def process_data_pipeline(raw_data :pd.DataFrame, num_feat:'list of numbers', categ_feat :'list of strings' = None, categ_feat_vals:'list of strings' = None, just_transform :bool = False, just_pipeline :bool = False):
    num_pipeline = Pipeline([
            ('feat_sel', FeatureSelector(num_feat, True)),
            ('Grade', FeatureCreator(['OverallCond', 'OverallQual'], lambda x, y: x / y, as_dataframe=True, feat_name='Grade')),
            ('Age', FeatureCreator(['YrSold', 'YearBuilt'], lambda x,y: x - y, as_dataframe=True, feat_name='Age')),
            ('RemodAge', FeatureCreator(['YrSold', 'YearRemodAdd'], lambda x,y: x - y, as_dataframe=True, feat_name='RemodAge')),
            ('TotalSF', FeatureCreator(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], lambda x,y: x + y, as_dataframe=True, feat_name='TotalSF')),
            ('drop_cat_feat', FeatureDropper(['YrSold', 'OverallCond'], as_dataframe=True)),
            ('imputer_mean', Imputer(strategy='mean')),
            ('std_scaler', RobustScaler())
        ]) 
    if categ_feat is None:
        if just_transform is True:
            return num_pipeline.transform(raw_data)
        return num_pipeline.fit_transform(raw_data)

    categ_cols = [raw_data[col].unique() for col in categ_feat] if categ_feat_vals is None else categ_feat_vals

    cat_pipeline = Pipeline([
            ('feat_sel', FeatureSelector(categ_feat, True)),
            ('imputer_most_frequent', CategoricalImputer()),
            ('encode', OneHotEncoder(sparse=False) if categ_cols is None else OneHotEncoder(categories=categ_cols, sparse=False)),
        ])
    feat_union = FeatureUnion(transformer_list=[
            ('num_features', num_pipeline),
            ('cat_features', cat_pipeline),
        ])

    if just_pipeline is True:
        return feat_union

    if just_transform is True:
        return feat_union.transform(raw_data)
    return feat_union.fit_transform(raw_data)


# Cross-validate the given model to the data provided and tune its hyperparameters, print the score and return the best estimator
def find_best_estimator(model, hyperparameters, data, labels, cv :int = 5):
    grid_search = GridSearchCV(model, hyperparameters, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(data, labels)

    print(np.sqrt(-grid_search.best_score_), grid_search.best_params_)

    return grid_search.best_estimator_


def main():
    # Load data and run brief analysis on it
    raw_data = load_data('train.csv')
    quick_analysis(raw_data)

    plt.hist(raw_data['SalePrice'])
    plt.show() 

    # View all unique values of categorical features
    non_numeric_cols = raw_data.loc[:, raw_data.dtypes == object]

    for col in non_numeric_cols.columns:
        print(non_numeric_cols[col].value_counts())

    # Analize correlations between features and the label
    corr_matrix = raw_data.corr()
    sale_correl = corr_matrix['SalePrice'].sort_values(ascending=False)
    print(sale_correl)

    # Feature engineering the following:
    #   Grade = OverallQual / OverallCond
    #   Age = YrSold - YearBuilt
    #   RemodAge = YrSold - YearRemodAdd
    #   TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF

    raw_data['Grade'] = raw_data['OverallQual'] / raw_data['OverallCond']
    raw_data['Age'] = raw_data['YrSold'] - raw_data['YearBuilt'] 
    raw_data['RemodAge'] = raw_data['YrSold'] - raw_data['YearRemodAdd']
    raw_data['TotalSF'] = raw_data['TotalBsmtSF'] + raw_data['1stFlrSF'] + raw_data['2ndFlrSF']

    # Correlation matrix for the new features
    corr_matrix = raw_data.corr()
    sale_correl = corr_matrix['SalePrice'].sort_values(ascending=False)
    print(sale_correl)

    # Check correlation of new features with their respective components
    age_correl = corr_matrix['Age'].sort_values(ascending=False)
    print('Age correlations:', age_correl, '\n')

    remod_age_correl = corr_matrix['RemodAge'].sort_values(ascending=False)
    print('RemodAge correlations:', remod_age_correl, '\n')

    grade_correl = corr_matrix['Grade'].sort_values(ascending=False)
    print('Grade correlations:', grade_correl, '\n')

    totalsf_correl = corr_matrix['TotalSF'].sort_values(ascending=False)
    print('TotalSF correlations:', totalsf_correl, '\n')

    # Correlation matrix vizualization
    corr_plot(raw_data, 'SalePrice', fig_size=(4, 4))
    corr_plot(raw_data, 'SalePrice', plot_type='hist', fig_size=(4, 4)) 
    
    # Change type of columns to reflect their nature. Concretely, change the YrSold, MoSold, MSZoning and OverallCond features to categorical ones
    raw_data['YrSold_C'] = raw_data['YrSold'].copy().astype(str)
    raw_data['MoSold'] = raw_data['MoSold'].astype(str)
    raw_data['MSZoning'] = raw_data['MSZoning'].astype(str)
    raw_data['OverallCond_C'] = raw_data['OverallCond'].copy().astype(str)

    num_cols = [
        'OverallQual',
        'OverallCond',
        'YearBuilt',
        'YearRemodAdd',
        'TotalBsmtSF',
        '1stFlrSF',
        '2ndFlrSF',
        'GarageCars',
        'GarageArea',
        'FullBath',
        'YrSold',
    ]
    cat_cols = [
        'MSZoning',
        'Street',
        'Utilities',
        'Neighborhood',
        'ExterQual',
        'ExterCond',
        'BsmtQual',
        'BsmtCond',
        'Heating',
        'CentralAir',
        'PavedDrive',
        'SaleType',
        'SaleCondition',
        'YrSold_C',
        'MoSold',
        'OverallCond_C',
    ]

    # Create a list of all values that the categorical features can take
    cat_cols_categs = [raw_data[col].unique() for col in cat_cols]
    print(cat_cols_categs)
    
    # Create the pipeline to process data
    num_pipeline = Pipeline([
            ('feat_sel', FeatureSelector(num_cols, True)),
            ('Grade', FeatureCreator(['OverallCond', 'OverallQual'], lambda x, y: x / y, as_dataframe=True, feat_name='Grade')),
            ('Age', FeatureCreator(['YrSold', 'YearBuilt'], lambda x,y: x - y, as_dataframe=True, feat_name='Age')),
            ('RemodAge', FeatureCreator(['YrSold', 'YearRemodAdd'], lambda x,y: x - y, as_dataframe=True, feat_name='RemodAge')),
            ('TotalSF', FeatureCreator(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], lambda x,y: x + y, as_dataframe=True, feat_name='TotalSF')),
            ('drop_cat_feat', FeatureDropper(['YrSold', 'OverallCond'], as_dataframe=True)),
            ('imputer_mean', Imputer(strategy='mean')),
            ('std_scaler', RobustScaler())
        ]) 

    cat_pipeline = Pipeline([
            ('feat_sel', FeatureSelector(cat_cols, True)),
            ('imputer_most_frequent', CategoricalImputer()),
            ('encode', OneHotEncoder(categories=cat_cols_categs, sparse=False)),
        ])
    feat_union = FeatureUnion(transformer_list=[
            ('num_features', num_pipeline),
            ('cat_features', cat_pipeline),
        ])

    # Create the train data and labels
    train_labels = raw_data['SalePrice'].copy()
    train_feat = feat_union.fit_transform(raw_data)

    # Check the linear regression model 
    lin_reg = LinearRegression()
    print('Linear regression best hyperparameters:')
    final_lr_model = find_best_estimator(lin_reg, [{}], train_feat, train_labels) 

    # Check the decision tree model
    hyperparams_vals = [
        {'max_features': [6, 10, 12, 16, 18, 20, 24]},
    ]
        
    dt_reg = DecisionTreeRegressor(random_state=42)
    print('Decision tree best hyperparameters:')
    final_dt_model = find_best_estimator(dt_reg, hyperparams_vals, train_feat, train_labels) 

    # Check the random forest model
    hyperparams_vals = [
        {'n_estimators': [200, 225, 250], 'max_features': [16, 24, 30]},
        {'bootstrap': [False], 'n_estimators': [220, 225], 'max_features': [24, 28]},
    ]

    forest_reg = RandomForestRegressor(n_jobs=-1, random_state=42)
    print('Random forest best hyperparameters:')
    final_rf_model = find_best_estimator(forest_reg, hyperparams_vals, train_feat, train_labels)

    # Check the XGBoost model
    hyperparams_vals = [
        {'n_estimators': [450, 500, 400], 'max_features': [2, 4, 8], 'max_depth': [3, 4, None]},
    ]

    xgbr_reg = XGBRegressor(learning_rate=0.05, n_threads=-1, random_state=42)
    print('XGBoost regressor best hyperparameters:')
    final_xgb_model = find_best_estimator(xgbr_reg, hyperparams_vals, train_feat, train_labels)

    # Check the SVM model
    hyperparams_vals = [
        {'kernel': ['linear', 'sigmoid', 'rbf'], 'gamma': ['auto', 'scale']},
        {'kernel': ['poly'], 'gamma': ['auto', 'scale'], 'degree': [3, 4, 5]},
    ]

    svm_reg = SVR()
    print('Support vector machine best hyperparameters:')
    final_svm_model = find_best_estimator(svm_reg, hyperparams_vals, train_feat, train_labels)

    # Check the ElasticNet model
    hyperparams_vals = [
        {'alpha': [0.0005, 0.005, 0.05, 0.2], 'l1_ratio': [0.1, 0.25, 0.75, 0.9]},
    ]

    enet_reg = ElasticNet(max_iter=100000000, tol=0.001)
    print('ElasticNet best hyperparameters:')
    final_enet_model = find_best_estimator(enet_reg, hyperparams_vals, train_feat, train_labels)

    # Check the feature importances for both random forest algorithms
    rf_feat_imp = final_rf_model.feature_importances_
    xgb_feat_imp = final_xgb_model.feature_importances_

    other_feat = ['Grade', 'RemodAge', 'TotalSF']
    all_features = num_cols.copy()
    print(num_cols)
    for cat_values in cat_cols_categs.copy():
        all_features.extend(cat_values)
    all_features.extend(other_feat.copy())

    print('Random forest feature importances:')
    for feat in sorted(zip(rf_feat_imp, all_features), reverse=True):
        print(feat)

    print('\nXGBoost feature importances:')
    for feat in zip(xgb_feat_imp, all_features):
        print(feat)

    # Load and process test data
    test_data = load_data('test.csv')
    test_data['YrSold_C'] = test_data['YrSold'].copy().astype(str).replace('nan', None)
    test_data['MoSold'] = test_data['MoSold'].astype(str).replace('nan', None)
    test_data['MSZoning'] = test_data['MSZoning'].astype(str).replace('nan', None)
    test_data['OverallCond_C'] = test_data['OverallCond'].copy().astype(str).replace('nan', None)
    test_feat = feat_union.transform(test_data)

    # Predict using the combination of Random Forest and XGBoost
    rf_predictions = final_rf_model.predict(test_feat)
    xgb_predictions = final_xgb_model.predict(test_feat)
    predictions = rf_predictions * 0.35 + xgb_predictions * 0.65

    # Save resulting predictions
    pred_df = pd.DataFrame()
    pred_df['Id'] = test_data['Id']
    pred_df['SalePrice'] = predictions.flatten()

    print(pred_df)
    pred_df.to_csv('submission_rf_xgb.csv')

    # Predict using only the XGBoost model
    xgb_predictions = final_xgb_model.predict(test_feat)
    predictions = xgb_predictions.copy() 

    pred_df = pd.DataFrame()
    pred_df['Id'] = test_data['Id']
    pred_df['SalePrice'] = predictions.flatten()

    print(pred_df)
    pred_df.to_csv('submission_xgb.csv')

    
if __name__ == '__main__':
    main()


