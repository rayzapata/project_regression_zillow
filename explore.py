#Z0096


# import from python libraries and modules
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# import visual tools
import matplotlib.pyplot as plt
import seaborn as sns

# import from created modules
from prepare import prepare_mvp, split_data


#################### Explore Data ####################


def get_tax_rates(use_csv=True):
    '''

    Obtain tax rates for each property with their county name, and
    output those distributions to a plot and dataframe to return

    '''

    # obtain prepared data to calcualte tax rates
    df = prepare_mvp(use_csv=use_csv)
    # convert into tax rate DataFrame
    df['tax_rate'] = df.tax_amount_usd / df.tax_value_usd
    df = df[['county', 'tax_amount_usd', 'tax_value_usd', 'tax_rate']].sort_values('county')
    # set figure dimensions for plot
    plt.figure(figsize=(30,15))
    # start plot
    plt.hist(df[df.county == 'Los Angeles'].tax_rate, bins=np.linspace(0, 0.1, 50), color='red', alpha=0.25, log=True, label='Los Angeles')
    plt.hist(df[df.county == 'Orange'].tax_rate, bins=np.linspace(0, 0.1, 50), color='green', alpha=0.25, log=True, label='Orange')
    plt.hist(df[df.county == 'Venture'].tax_rate, bins=np.linspace(0, 0.1, 50), color='blue', alpha=0.25, log=True, label='Venture')
    plt.rcParams['legend.title_fontsize'] = 20
    plt.xlim(0, 0.1)
    plt.xlabel('Tax Rate')
    plt.title('Distributions of Tax Rates for Each County')
    plt.legend(title='County')
    plt.show()
    
    return df


def get_tax_rates_county(use_csv=True):
    '''

    Takes in Zillow DataFrame with county names, property tax values,
    and property tax payments and adds their means to a DataFrame with
    a calculate tax rate per county.

    '''

    # obtain prepared data to get tax rates
    df = prepare_mvp(use_csv=use_csv)
    # assign blank DataFrame to append to
    df_taxes = pd.DataFrame()
    # start loop for each county in DataFrame
    for county_name in df.county.sort_values().unique():
        # set mask to filter values per county
        subset = df[df.county == county_name]
        # append dictionary to DataFrame values for each county
        df_taxes = df_taxes.append({
            'county_name': county_name,
            'avg_tax_amount_usd': subset.tax_amount_usd.mean(),
            'avg_tax_value_usd': subset.tax_value_usd.mean(),
            'tax_rate': (subset.tax_amount_usd / subset.tax_value_usd).mean()
            }, ignore_index=True)
    df = df_taxes.set_index('county_name')

    return df


def get_tax_rate():
    '''
    '''

    # obtain prepared data to calcualte tax rates
    df = prepare_mvp(use_csv=use_csv)
    # convert into tax rate DataFrame
    df['tax_rate'] = df.tax_amount_usd / df.tax_value_usd
    df = df[['county', 'tax_amount_usd', 'tax_value_usd', 'tax_rate']].sort_values('county')
    # set figure dimensions for plot
    plt.figure(figsize=(30,15))





def explore_mvp(use_csv=True):
    '''

    Bringing in MVP prepared data, this function splits it into X, y for train,
    validate, and test using tax_value_usd as our target variable. This
    functions returns the six splits as well as a DataFrame for exploration
    containing only the 60% the training data

    '''

    # obtain prepared data and drop non-mvp column
    df = prepare_mvp(use_csv=use_csv)
    df = df.drop(columns=['county', 'tax_amount_usd'])
    # split data into appropraite DataFrames
    df, \
    X_train, y_train, \
    X_validate, y_validate, \
    X_test, y_test = split_data(df, 'tax_value_usd')
    
    return (df,
            X_train, y_train,
            X_validate, y_validate,
            X_test, y_test)



#################### Visualize Data ####################


def plot_heat(df, target):
    
    '''

    Use seaborn to create heatmap with coeffecient annotations to
    visualize correlation between all variables


    '''

    n_vars = len(list(df))
    # Set up large figure size for easy legibility
    plt.figure(figsize=(n_vars + 5, n_vars + 1))
    # assign pd.corr() output to variable and create a mask to remove
    # redundancy from graphic
    corr = df.corr()
    mask = np.triu(corr, k=0)
    # define custom cmap for heatmap where the darker the reds the more
    # positive and vice versa for blues
    cmap = sns.diverging_palette(h_neg=220, h_pos=13, sep=25, as_cmap=True)
    # create graphic with zero centered cmap and annotations set to one
    # significant figure
    sns.heatmap(corr, cmap=cmap, center=0, annot=True, fmt=".1g", square=True,
                mask=mask, cbar_kws={
                                     'shrink':0.5,
                                     'aspect':50,
                                     'use_gridspec':False,
                                     'anchor':(-0.75,0.75)
                                      })
    # format xticks for improved legibility and clarity
    plt.xticks(ha='right', va='top', rotation=35, rotation_mode='anchor')
    plt.title('Correlation Heatmap')
    plt.show()


def plot_univariate(data, variable):
    '''

    This function takes the passed DataFrame the requested and plots a
    configured boxenplot and distrubtion for it side-by-side

    '''

    # set figure dimensions
    plt.figure(figsize=(30,8))
    # start subplot 1 for boxenplot
    plt.subplot(1, 2, 1)
    sns.boxenplot(x=variable, data=data)
    plt.axvline(data[variable].median(), color='pink')
    plt.axvline(data[variable].mean(), color='red')
    plt.xlabel('')
    plt.title('Enchanced Box Plot', fontsize=25)
    # start subplot 2 for displot
    plt.subplot(1, 2, 2)
    sns.histplot(data=data, x=variable, element='step', kde=True, color='cyan',
                                line_kws={'linestyle':'dashdot', 'alpha':1})
    plt.axvline(data[variable].median(), color='pink')
    plt.axvline(data[variable].mean(), color='red')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Distribution', fontsize=20)
    # set layout and show plot
    plt.suptitle(f'{variable} $[n = {data[variable].count():,}]$', fontsize=25)
    plt.tight_layout()
    plt.show()


def plot_discrete_to_continous(data, discrete_var, continous_var, swarm_n=2000,
                                            r_type='pearson', random_state=19):
    '''

    Takes in a DataFrame and lists of discrere and continuous variables and
    plots a boxenplot, swarmplot, and regplot for each against the other,
    providing either the pearson (default) or spearman r measurement in the
    title

    '''

    # choose coefficient
    if r_type == 'pearson':
        r = pearsonr(data[discrete_var], data[continous_var])[0]
    elif r_type =='spearman':
        r = spearmanr(data[discrete_var], data[continous_var])[0]
    # set figure dimensions
    plt.figure(figsize=(30,10))
    # start subplot 1 for boxplot
    plt.subplot(1, 3, 1)
    sns.boxenplot(x=discrete_var, y=continous_var, data=data)
    plt.xlabel('')
    plt.ylabel(f'{continous_var}', fontsize=20)
    # start subplot 2 for boxplot
    plt.subplot(1, 3, 2)
    sns.swarmplot(x=discrete_var, y=continous_var, data=data.sample(n=swarm_n,
                                                    random_state=random_state))
    plt.xlabel(f'{discrete_var}', fontsize=20)
    plt.ylabel('')
    # start subplot 3 for boxplot
    plt.subplot(1, 3, 3)
    sns.regplot(x=discrete_var, y=continous_var, data=data, marker='*',
                                                    line_kws={'color':'red'})
    plt.xlabel('')
    plt.ylabel('')
    # set title for graphic and output
    plt.suptitle(f'{discrete_var} to {continous_var} $[r = {r:.2f}]$',
                                                                fontsize=25)
    plt.tight_layout()
    plt.show()


def plot_joint(data, x, y, r_type='pearson'):
    '''

    Takes in a DataFrame and the specified x, y variables and plots a
    configured joint plot with the pearson (default) or spearman r measurement
    in the title

    '''

    # choose coefficient
    if r_type == 'pearson':
        r = pearsonr(data[x], data[y])[0]
    elif r_type =='spearman':
        r = spearmanr(data[x], data[y])[0]
    # plot jointplot of continuous variables
    sns.jointplot(x, y, data, kind='reg', height=10,
                  joint_kws={'marker':'+', 'line_kws':{'color':'red'}},
                  marginal_kws={'color':'cyan'})
    # set labels for x, y axes
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    # set title of compared variables
    plt.suptitle(f'{x} to {y} $[r = {r:.2f}]$')
    plt.tight_layout()
    # show plot
    plt.show()


def corr_test(data, x, y, alpha=0.05, r_type='pearson'):
    '''

    Performs a pearson or spearman correlation test and returns the r
    measurement as well as comparing the return p valued to the pass or
    default significance level, outputs whether to reject or fail to
    reject the null hypothesis
    
    '''
    
    # obtain r, p values
    if r_type == 'pearson':
        r, p = pearsonr(data[x], data[y])
    if r_type == 'spearman':
        r, p = spearmanr(data[x], data[y])
    # print reject/fail statement
    print(f'''{r_type:>10} r = {r:.2g}
+--------------------+''')
    if p < alpha:
        print(f'''
        Due to p-value {p:.2g} being less than our significance level of \
{alpha}, we must reject the null hypothesis 
        that there is not a linear correlation between "{x}" and "{y}."
        ''')
    else:
        print(f'''
        Due to p-value {p:.2g} being greater than our significance level of \
{alpha}, we fail to reject the null hypothesis 
        that there is not a linear correlation between "{x}" and "{y}."
        ''')
