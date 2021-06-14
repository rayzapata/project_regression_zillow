#Z0096


# import from python libraries and modules
import pandas as pd
import numpy as np
from os.path import isfile

# import functions from created modules
from env import get_connection


#################### Acquire Data ####################


def get_sql(query, db_name, use_csv=True):
    '''

    Takes SQL query and database name as `string` and runs through
    pandas read_sql function, using get_connection from env.py.
    Requires Codeup database login credentials.

    use_csv=True will use data from existing CSV file if one exists,
    default behavior

    use_csv=False will obtain new query return and overwrite existing
    CSV if one exists

    '''

    # check if CSV file already exists or use_csv=False
    if use_csv == False or isfile(f'{db_name}.csv') == False:
        # acquire data from database
        df = pd.read_sql(query, get_connection(db_name))
        # write acquired data to CSV file
        df.to_csv(f'{db_name}.csv', index=False)
        # drop any existing duplicates
        df.drop_duplicates()
    else:
        # read into DataFrame from CSV file
        df = pd.read_csv(f'{db_name}.csv')

    return df


def acquire_mvp(use_csv=False):
    '''

    Using get_sql function we pass a specific query to obtain the MVP'
    variables to create a DataFrame using the property_id unique
    identifier as the index

    This function acquires only the data need to construct the MVP

    '''

    # assign sql query for specified data
    query = '''
            SELECT
                properties_2017.id AS property_id,
                bedroomcnt AS bedrooms,
                bathroomcnt AS bathrooms,
                fips,
                calculatedfinishedsquarefeet AS square_feet,
                taxamount AS tax_amount_usd,
                taxvaluedollarcnt AS tax_value_usd
            FROM properties_2017
            INNER JOIN predictions_2017 USING(parcelid)
            LEFT JOIN propertylandusetype USING(propertylandusetypeid)
            WHERE
                propertylandusetypeid IN (261, 263, 264, 266, 268, 275, 276, 279) AND
                CAST(transactiondate AS DATE) BETWEEN 20170501 AND 20170831
            ORDER BY
                properties_2017.id ASC
            ;'''
    # use get_sql function to read into DataFrame
    df = get_sql(query, 'zillow', use_csv=use_csv)
    # set id to index
    df = df.set_index('property_id')
    # drop rows with null values
    df = df.dropna()

    return df
