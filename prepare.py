#Z0096


# import from python libraries and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import scipy.stats as stats

# import data manipulation tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# import from created modules
from acquire import acquire_mvp


#################### Prepare Data ####################


def shed_zscore_outliers(df, exclude=None):
    '''
    
    Takes in DataFrame and removed rows with values that are more than
    three standard deviations from the mean. 
    
    exclude=None will pass all columns into check, default behavior
    
    Can pass column name or list of column names as 'strings' to
    exclude from checking z-scores
    
    '''
    
    # check if any columns are to be excluded
    if exclude == None:
        # reassign DataFrame with all values more than 3 standard
        # deviations from mean
        df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
        return df
    else:
        # assign DataFrame with non-excluded values more than 3
        # standard deviations from mean
        df = df[(np.abs(
            stats.zscore(df.loc[:, df.columns != exclude])) < 3).all(axis=1)]
        return df


def make_dummies(df, cols):
    '''
    '''

    # create dum-dums from passed cols list, drop first
    df = pd.get_dummies(df, columns=cols, drop_first=True)

    return df


def split_data(df, target):
    '''
    '''

    # assign X, y
    X = df.drop(columns=target)
    y = pd.DataFrame(df[target])
    # split X data into train, validate, and test datasets
    X_train_validate, X_test = train_test_split(X, test_size=0.2, 
                                            random_state=19)
    X_train, X_validate = train_test_split(X_train_validate, test_size=0.25,
                                            random_state=19)
    # split y data into train, validate, and test datasets
    y_train_validate, y_test = train_test_split(y, test_size=0.2, 
                                            random_state=19)
    y_train, y_validate = train_test_split(y_train_validate, test_size=0.25,
                                            random_state=19)
    # create explore DataFrame using train data
    df = pd.concat((X_train, y_train), axis=1)

    return (df,
            X_train, y_train,
            X_validate, y_validate,
            X_test, y_test)


def prepare_mvp(use_csv=False):
    '''
    '''

    # acquire mvp data from database
    df = acquire_mvp(use_csv=use_csv)
    # remove outliers more than 3 stdev from mean
    df = shed_zscore_outliers(df, exclude='fips')
    # remove 0 values from bedrooms and bathrooms
    df = df.loc[((df.bedrooms != 0) & (df.bathrooms != 0))]
    # replace fips numerical codes with string county names and rename column
    df.fips = np.where(df.fips == 6037.0, 'Los Angeles', df.fips)
    df.fips = np.where(df.fips == '6059.0', 'Orange', df.fips)
    df.fips = np.where(df.fips == '6111.0', 'Ventura', df.fips)
    df = df.rename(columns=({'fips':'county'}))

    return df


#################### Visualize Data ####################


def show_distributions(df):
    '''
    '''

    # define dimensions for subplots
    n_rows = ceil(len(list(df)) / 4)
    n_cols = 4
    n_plot = 0
    # define figure dimensions
    plt.figure(figsize=(40, n_rows * 10))
    # start loop for plotting variables
    for col in list(df):
        n_plot += 1
        plt.subplot(n_rows, n_cols, n_plot)
        plt.hist(x=df[col])
        plt.title('')
        plt.xlabel(col)
        plt.xscale('linear')
    # set figure title
    plt.suptitle('Distribution of Variables')
    plt.show()
