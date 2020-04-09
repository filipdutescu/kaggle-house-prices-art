import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker as tick

import transformers as trans

from sklearn.pipeline import Pipeline 


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


def main():
    no_data_cols = [ 'Id', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature' ]
    raw_data = load_data('train.csv', no_data_cols)
    raw_data.dropna(inplace=True)
    #quick_analysis(raw_data)

    #plt.hist(raw_data['SalePrice'])
    #plt.show()

    #non_numeric_cols = raw_data.loc[:, raw_data.dtypes == object]
    #quick_analysis(non_numeric_cols)

    #for col in non_numeric_cols.columns:
    #    print(non_numeric_cols[col].value_counts())
    #    input()

    #raw_data.drop(columns=[ 'YearBuilt', 'YearRemodAdd' ], inplace=True)
    #raw_data['PercUnfBsmt'] = raw_data['BsmtUnfSF'] / raw_data['TotalBsmtSF'] 

    new_feat_pipeline = Pipeline([
            ('Grade', trans.FeatureCreator(['OverallCond', 'OverallQual'], lambda x, y: x / y, as_dataframe=True, feat_name='Grade')),
            ('Age', trans.FeatureCreator(['YrSold', 'YearBuilt'], lambda x,y: x - y, as_dataframe=True, feat_name='Age')),
            ('RemodAge', trans.FeatureCreator(['YrSold', 'YearRemodAdd'], lambda x,y: x - y, as_dataframe=True, feat_name='RemodAge')),
            ('TotalSF', trans.FeatureCreator(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], lambda x,y: x + y, as_dataframe=True, feat_name='TotalSF')),
            ]) 
    raw_data = new_feat_pipeline.fit_transform(raw_data)

    corr_matrix = raw_data.corr()
    sale_correl = corr_matrix['SalePrice'].sort_values(ascending=False)
    print(sale_correl)

    sale_str_corr = sale_correl.where(abs(sale_correl) > 0.5).where(abs(sale_correl) < 1).dropna() 
#    print(sale_str_corr)

#    corr_plot(raw_data, 'SalePrice', fig_size=(4, 4))
#    corr_plot(raw_data, 'SalePrice', y_lower_scale=False, same_fig=False)
    plt.hist(x=raw_data['SalePrice'])
    plt.show()
    corr_plot(raw_data, 'SalePrice', plot_type='hist', y_lower_scale=False, same_fig=False)

    
if __name__ == '__main__':
    main()


