#Z0096


# import from python libraries and modules
import pandas as pd
import numpy as np

#import visualization tools
import matplotlib.pyplot as plt

# import modeling tools
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler


#################### Create & Test Models ####################


def train_model(X, y, model, model_name):
    '''

    Takes in the X_train and y_train, model object and model name, fit the
    model and returns predictions and a dictionary containg the model RMSE
    and R^2 scores on train

    '''

    # fit model to X_train_scaled
    model.fit(X, y)
    # predict X_train
    predictions = model.predict(X)
    # get rmse and r^2 for model predictions on X
    rmse, r2 = get_metrics(y, predictions)
    performance_dict = {'model':model_name, 'RMSE':rmse, 'R^2':r2}
    
    return predictions, performance_dict


def model_testing(X, y, model, model_name):
    '''

    Takes in the X and y for validate or test, model object and model name and
    returns predictions and a dictionary containg the model RMSE and R^2 scores
    on validate or test

    '''
    
    # obtain predictions on X
    predictions = model.predict(X)
    # get for performance and assign them to dictionary
    rmse, r2 = get_metrics(y, predictions)
    performance_dict = {'model':model_name, 'RMSE':rmse, 'R^2':r2}
    
    return predictions, performance_dict


#################### Scale Data #########################


def minmax(X_train, X_validate, X_test, features_to_scale=None):
    '''

    Takes in the X for train, validate, and test and an option list and scales
    all or the list of features using the minmax scaler with default setting,
    outputs dataframes with all or only list variables scaled

    '''
    
    # check if list of specific features are passed
    if features_to_scale == None:
        # assign list of all columns is no list is passed
        features_to_scale = list(X_train)
    # create sacler object
    scaler = MinMaxScaler()
    # fit scaler to X_train
    scaler.fit(X_train[features_to_scale])
    # transform X_train and create new DataFrame for scaled data
    X_train_scaled = scaler.transform(X_train[features_to_scale])
    X_train_scaled = pd.DataFrame(X_train_scaled,
                            columns=features_to_scale).set_index(X_train.index)
    # combine scaled and unscaled features if list was passed
    if features_to_scale != None:
        X_train_scaled = pd.concat((X_train.drop(columns=features_to_scale),
                            X_train_scaled), axis=1)
    # transform X_validate and create new DataFrame for scaled data
    X_validate_scaled = scaler.transform(X_validate[features_to_scale])
    X_validate_scaled = pd.DataFrame(X_validate_scaled,
                        columns=features_to_scale).set_index(X_validate.index)
    # combine scaled and unscaled features if list was passed
    if features_to_scale != None:
        X_validate_scaled = pd.concat((X_validate.drop(columns=features_to_scale),
                        X_validate_scaled), axis=1)
    # transform X_test and create new DataFrame for scaled data
    X_test_scaled = scaler.transform(X_test[features_to_scale])
    X_test_scaled = pd.DataFrame(X_test_scaled,
                            columns=features_to_scale).set_index(X_test.index)
    # combine scaled and unscaled features if list was passed
    if features_to_scale != None:
        X_test_scaled = pd.concat((X_test.drop(columns=features_to_scale),
                            X_test_scaled), axis=1)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled


#################### Explore Features ####################


def select_kbest(X, y, k=1, score_func=f_regression):
    '''

    Takes in the X, y train and an optional k values and score_func to use
    SelectKBest to return k (default=1) best variables for predicting the
    target of y
    
    '''

    # assign SelectKBest using f_regression and top two features default
    selector = SelectKBest(score_func=score_func, k=k)
    # fit selector to training set
    selector.fit(X, y)
    # assign and apply mask to DataFrame for column names
    mask = selector.get_support()
    top_k = X.columns[mask].to_list()
    return top_k


def select_rfe(X, y, n=1, model=LinearRegression(), rank=False):
    '''

    Takes in the X, y train and an optional n values and model to use with
    RFE to return n (default=1) best variables for predicting the
    target of y, optionally can be used to output ranks of features in
    predictions

    '''

    # assign RFE using LinearRegression and top two features as default
    selector = RFE(estimator=model, n_features_to_select=n)
    # fit selector to training set
    selector.fit(X, y)
    # assign and apply mask to DataFrame for column names
    mask = selector.get_support()
    top_n = X.columns[mask].to_list()
    # check if rank=True
    if rank == True:
        # print DataFrame of rankings
        print(pd.DataFrame(X.columns, selector.ranking_,
                           [f'n={n} RFE Rankings']).sort_index())
    return top_n


#################### Model Performance ####################


def get_metrics(true, predicted, display=False):
    '''

    Takes in the true and predicted values and returns the rmse and r^2 for the
    model performance

    '''
    
    rmse = mean_squared_error(true, predicted) ** 0.5
    r2 = explained_variance_score(true, predicted)
    if display == True:
        print(f'Model RMSE: {rmse:.2g}')
        print(f'       R^2: {r2:.2g}')
    return rmse, r2


def plot_residuals(y_true, y_predicted):
    '''

    Takes in 1 to 4 prediction sets and returns a configured scatterplot of the
    residual errors of those predictions against the passed true set
    
    '''

    # set figure dimensions
    plt.figure(figsize=(60, 40))
    plt.rcParams['legend.title_fontsize'] = 50
    # scatterplot for each up to four predictions passed in list
    plt.scatter(y_true, (y_predicted.iloc[0:,0] - y_true), alpha=1,
                    color='cyan', s=250, label=y_predicted.iloc[0:,0].name,
                    edgecolors='black')
    if len(y_predicted.columns) > 1:
        plt.scatter(y_true, (y_predicted.iloc[0:,1] - y_true), alpha=0.75, 
                    color='magenta', s=250, label=y_predicted.iloc[0:,1].name, 
                    edgecolors='black')
    if len(y_predicted.columns) > 2:
        plt.scatter(y_true, (y_predicted.iloc[0:,2] - y_true), alpha=0.75, 
                    color='yellow', s=250, label=y_predicted.iloc[0:,2].name, 
                    edgecolors='black')
    if len(y_predicted.columns) > 3:
        plt.scatter(y_true, (y_predicted.iloc[0:,3] - y_true), alpha=0.5, 
                    color='black', s=250, label=y_predicted.iloc[0:,3].name, 
                    edgecolors='white')
    if len(y_predicted.columns) > 4:
        return 'Can only plot up to four models\' predictions'
    # add zero line for ease of readability
    plt.axhline(label='', color='red', linewidth=5, linestyle='dashed',
                    alpha=0.25)
    # model legend
    plt.legend(title='Models', loc=(0.025,0.05), fontsize=50)
    
    # set labels and title
    plt.xlabel('\nTrue Value\n', fontsize=50)
    plt.ylabel('\nPredicted Value Error\n', fontsize=50)
    plt.title(f'\nPrediction Residuals of {y_true.name}\n', fontsize=50)
    # show plot
    plt.show()
