# Predicting Home Values with Zillow
![](https://github.com/ray-zapata/project_regression_zillow/blob/main/assets/logo.png)

### Table of Contents
---

I.   [Project Overview             ](#i-project-overview)
1.   [Description                  ](#1-description)
2.   [Deliverables                 ](#2-deliverables)

II.  [Project Summary              ](#ii-project-summary)
1.   [Goals                        ](#1-goals)
2.   [Initial Thoughts & Hypothesis](#2-initial-thoughts--hypothesis)
3.   [Findings & Next Phase        ](#3-findings--next-phase)

III. [Data Context                 ](#iii-data-context)
1.   [Database Relationships       ](#1-database-relationships)
2.   [Data Dictionary              ](#2-data-dictionary)

IV.  [Process                      ](#iv-process)
1.   [Project Planning             ](#1-project-planning)
2.   [Data Acquisition             ](#2-data-acquisition)
3.   [Data Preparation             ](#3-data-preparation)
4.   [Data Exploration             ](#4-data-exploration)
5.   [Modeling & Evaluation        ](#5-modeling--evaluation)
6.   [Product Delivery             ](#6-product-delivery)

V.   [Modules               ](#v-modules)

VI.  [Project Reproduction  ](#vi-project-reproduction)

![](https://github.com/ray-zapata/project_regression_zillow/blob/main/assets/divider.png)

### I. Project Overview
---

#### 1. Description

This project serves to fulfill the requests made to the Zillow data science department to predict the values of "single unit properties." With the available data, it is also requested to obtain county specific tax rates and their distributions.

#### 2. Deliverables

- GitHub repository and [README](#estimation-of-home-value-with-zillow) stating project overview, goals, findings, and summary
- Jupyter [Notebook](https://nbviewer.jupyter.org/github/ray-zapata/project_regression_zillow/blob/main/zillow.ipynb) showing detailed process through data science pipeline
- Slide deck [presentation](https://www.canva.com/design/DAEhaQ2Ce24/q9aP7PIiZ7IF6nbTNzyVcA/view?utm_content=DAEhaQ2Ce24&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink) summarizing findings of driver for single unite property values


### II. Project Summary
---

#### 1. Goals

Utilizing the data available from Zillow, this project sets out to best predict value in USD of properties sold in 2017 during the "hot months" of May, June, July, and August. Property value in this context is defined as the tax appraised value of land and any structures built upon it. Through the process of preparing and exploring data, we will find drivers of single unit property value and create predictive regression models to predict that value using the count of bathrooms, bedrooms and the calculated finished square feet of the structure.

Secondary to this main objective, the data science team is also requested to obtain the distributions and estimation of tax rates for each of the counties where these properties exist. Using the Federal Information Processing Standards (FIPS) codes to obtain county names, and the tax assessed value and tax amount, we will calculate and visualize this request.

#### 2. Initial Thoughts & Hypothesis

The initial hypothesis of the project was that as the dimensions of the structure on the property increased, so too would it's value. The same was considered true for the count of bathrooms and bedrooms on the property as well, but was not believed to be as strong a driver as size. It is also believed that size and location of the land itself would play a role in property value, and this is a potential avenue of future exploration after completing the minimally viable product (MVP).

#### 3. Findings & Next Phase

Through statistical testing, we found evidence that counts of bathrooms and bedrooms as well as structure finished square feet had a positive relationship with property value. In creating our models, we utilized all three requested features as part of training and fitting for the MVP, and created a second degree polynomial regression model which performed well enough to make it through to the testing dataset. With a root mean squared error of $271,768.12 on the final test, this model maintained consistent performance through each stage. With each model created, it was found that the higher the true value, the larger the residuals of predictions grew.

Using the FIPS codes within the data, these identifiers were used with data from [census.gov](https://www.census.gov/prod/techdoc/cbp/95-96cd/fips-st.pdf) to obtain counties for each property. Using the tax assessed value and the tax payment for each property, the distribution of tax rates were obtained, and then a calculated average tax rate for each county as stated below. All counties were located within the US state of California.

**County Estimated Tax Rates**

| County      | Average Tax | Average Value | Tax Rate |
|:------------|:-----------:|:-------------:|:--------:|
| Los Angeles | $5,159.40   | $406,432.95   | 1.38%    |
| Orange      | $5,631.82   | $483,927.33   | 1.21%    |
| Ventura     | $5,117.08   | $437,588.39   | 1.20%    |

With additional time, we will be able to explore further variables as model features. It is a continuing hypothesis that location and lot size will play a strong role in better predicting when the true value is higher.

### III. Data Context
---

#### 1. Database Relationships

The Codeup `zillow` SQL database contains twelve tables, nine of which have foreign key links with our primary table `properties_2017`: `airconditioningtype`, `architecturalstyletype`, `buildingclasstype`, `heatingorsystemtype`, `predictions_2017`, `propertylandusetype`, `storytype`, `typeconstructiontype`, and `unique_properties`. Each table is connected by a pointed arrow with the corresponding foreign keys that link them. Many of these tables are unused in this project due to missing values, and this database map serves only to define the database.

![](https://github.com/ray-zapata/project_regression_zillow/blob/main/assets/databasemap.png)

#### 2. Data Dictionary

Following acquisition and preparation of the initial SQL database, the DataFrames used in this project contain the following variables. Contained values are defined along with their respective data types.

| Variable          | Definition                                         | Data Type |
|:------------------|:--------------------------------------------------:|:---------:|
| bedrooms          | count of bedrooms on property                      | integer   |
| bathrooms         | count of bathrooms and half-bathrooms on property  | float     |
| county            | human readable name of county where property exists| object    |
| fips              | federal information processing standards codes     | integer   |
| id                | unique identifier for each property                | index     |
| square_feet       | total calculated square feet in property structure | float     |
| tax_amount_usd    | property taxes based on assessed value in USD      | float     |
| tax_value_usd *   | assessed value of property in USD                  | float     |

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  * Target variable

### IV. Process
---

#### 1. Project Planning
🟢 **Plan** ➜ ☐ _Acquire_ ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Build this README containing:
    - Project overview
    - Initial thoughts and hypotheses
    - Project summary
    - Instructions to reproduce
- [x] Build and use [Trello](https://trello.com/b/V0IPaIgF/regression-project) board for data science pipeline
- [x] Consider project needs versus project desires

#### 2. Data Acquisition
✓ _Plan_ ➜ 🟢 **Acquire** ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Obtain initial data and understand its structure
- [x] Remedy any inconsistencies, duplications, or structural problems within data
- [x] Perform data summation

#### 3. Data Preparation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ 🟢 **Prepare** ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Address missing or inappropriate values, including outliers
- [x] Plot distributions of variables
- [x] Encode categorical variables
- [x] Consider and create new features as needed
- [x] Split data into `train`, `validate`, and `test`

#### 4. Data Exploration
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ 🟢 **Explore** ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Visualize relationships of variables
- [x] Formulate hypotheses
- [x] Perform statistical tests
- [x] Decide upon features and models to be used

#### 5. Modeling & Evaluation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ 🟢 **Model** ➜ ☐ _Deliver_

- [x] Establish baseline prediction
- [x] Create, fit, and predict with models
- [x] Evaluate models with out-of-sample data
- [x] Utilize best performing model on `test` data
- [x] Summarize, visualize, and interpret findings

#### 6. Product Delivery
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ ✓ _Model_ ➜ 🟢 **Deliver**
- [x] Prepare Jupyter Notebook of project details through data science pipeline
- [ ] With additional time, continue with exploration beyond MVP
- [x] Proof read and complete README and project repository
- [ ] Prepare slide deck presentation of project

### V. Modules
---

The created modules used in this project below contain full comments an docstrings to better understand their operation. Where applicable, all functions used `random_state=19` at all times. Use of functions requiring access to the Codeup database require an additional module named `env.py`. See project reproduction for more detail.

- [`acquire`](https://raw.githubusercontent.com/ray-zapata/project_regression_zillow/main/acquire.py): contains functions used in initial data acquisition leading into the prepare phase
- [`prepare`](https://raw.githubusercontent.com/ray-zapata/project_regression_zillow/main/prepare.py): contains functions used to prepare data for exploration and visualization
- [`explore`](https://raw.githubusercontent.com/ray-zapata/project_regression_zillow/main/explore.py): contains functions to visualize the prepared data and estimate the best drivers of property value
- [`model`  ](https://raw.githubusercontent.com/ray-zapata/project_regression_zillow/main/model.py): contains functions to create, test models and visualize their performance

### VI. Project Reproduction
---

To recreate and reproduce results of this project, you will need to create a module named `env.py`. This file will need to contain login credentials for the Codeup database server stored in their respective variables named `host`, `username`, and `password`. You will also need to create the following function within. This is used in all functions that acquire data from the SQL server to create the URL for connecting. `db_name` needs to be passed as a string that matches exactly with the name of a database on the server.

```py
def get_connection(db_name):
    return f'mysql+pymysql://{username}:{password}@{host}/{db_name}'
```

After its creation, ensure this file is not uploaded or leaked by ensuring git does not interact with it. When using any function housed in the created modules above, ensure full reading of comments and docstrings to understand its proper use and passed arguments or parameters.

[[Return to Top]](#predicting-home-values-with-zillow)
