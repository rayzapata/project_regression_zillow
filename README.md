# Predicting Home Values with Zillow
![](https://github.com/ray-zapata/project_regression_zillow/blob/main/assets/logo.png)

## Table of Contents

I.   [Project Overview      ](#i-project-overview)
1.   [Description           ](#1-description)
2.   [Deliverables          ](#2-deliverables)

II.  [Project Summary       ](#ii-project-summary)
1.   [Goals                 ](#1-goals)
2.   [Thoughts & Hypotheses ](#2-thoughts--hypotheses)
3.   [Findings & Next Phase ](#3-findings--next-phase)

III. [Data Context          ](#iii-data-context)
1.   [Database Relationships](#1-database-relationships)
2.   [Data Dictionary       ](#2-data-dictionary)

IV.  [Process               ](#iv-process)
1.   [Project Planning      ](#1-project-planning)
2.   [Data Acquisition      ](#2-data-acquisition)
3.   [Data Preparation      ](#3-data-preparation)
4.   [Data Exploration      ](#4-data-exploration)
5.   [Modeling & Evaluation ](#5-modeling--evaluation)
6.   [Product Delivery      ](#6-product-delivery)

V.   [Modules               ](#v-modules)

VI.  [Project Reproduction  ](#vi-project-reproduction)

![](https://github.com/ray-zapata/project_regression_zillow/blob/main/assets/divider.png)

## I. Project Overview

#### 1. Description

WIP

#### 2. Deliverables

WIP

## II. Project Summary

#### 1. Goals

WIP

#### 2. Thoughts & Hypotheses

WIP

#### 3. Findings & Next Phase

WIP

## III. Data Context

#### 1. Database Relationships

The Codeup `zillow` SQL database contains twelve tables, nine of which have foreign key links with our primary table `properties_2017`: `airconditioningtype`, `architecturalstyletype`, `buildingclasstype`, `heatingorsystemtype`, `predictions_2017`, `propertylandusetype`, `storytype`, `typeconstructiontype`, and `unique_properties`. Each table is connected by a pointed arrow with the corresponding foreign keys that link them. Many of these tables are unused in this project due to missing values, and this database map serves only to define the database.

![](https://github.com/ray-zapata/project_regression_zillow/blob/main/assets/databasemap.png)

#### 2. Data Dictionary

Following acquisition and preparation of the initial SQL database, the DataFrames used in this project contain the following variables. Contained values are defined along with their respective data types.

| Variable          | Definition                                         | Data Type |
|:-----------------:|:--------------------------------------------------:|:---------:|
| bedrooms          | count of bedrooms on property                      | integer   |
| bathrooms         | count of bathrooms and half-bathrooms on property  | float     |
| fips              | federal information processing standards codes     | integer   |
| square_feet       | total calculated square feet in property structure | float     |
| tax_amount_usd    | property taxes based on assessed value in USD      | float     |
| tax_value_usd *   | assessed value of property in USD                  | float     |

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  * Target variable

## IV. Process

#### 1. Project Planning
🟢 **Plan** ➜ ☐ _Acquire_ ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Build this [README](#estimation-of-home-value-with-zillow) containing:
    - Project overview
    - Initial thoughts and hypotheses
    - Project summary
    - Instructions to reporoduce
- [x] Build and use [Trello](https://trello.com/b/V0IPaIgF/regression-project) board for data science pipeline
- [x] Consider project needs versus project desires

#### 2. Data Acquisition
✓ _Plan_ ➜ 🟢 **Acquire** ➜ ☐ _Prepare_ ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Build [`acquire.py`](https://raw.githubusercontent.com/ray-zapata/project_regression_zillow/main/acquire.py) module
- [x] Obtain initial data and understand its structure
- [x] Plot distributions of variables
- [x] Perform data summation

#### 3. Data Preparation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ 🟢 **Prepare** ➜ ☐ _Explore_ ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Build [`prepare.py`](https://raw.githubusercontent.com/ray-zapata/project_regression_zillow/main/prepare.py) module
- [x] Address missing or inappropraite values, including outliers
- [x] Encode categorical variables
- [x] Consider and create new features as needed
- [x] Split data into `train`, `validate`, and `test`

#### 4. Data Exploration
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ 🟢 **Explore** ➜ ☐ _Model_ ➜ ☐ _Deliver_

- [x] Build [`explore.py`](https://raw.githubusercontent.com/ray-zapata/project_regression_zillow/main/explore.py) module
- [x] Visualize relationships of variables
- [x] Formulate hypotheses
- [x] Perform statistical tests
- [x] Decide features to use use for models

#### 5. Modeling & Evaluation
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ 🟢 **Model** ➜ ☐ _Deliver_

- [x] Establish baseline prediction
- [x] Establish features and models to be used
- [x] Create, fit, and predict with models
- [ ] Evaluate models with out-of-sample data
- [ ] Utilize best performing model on `test` data
- [ ] Summarize, visualize, and interpret findings

#### 6. Product Delivery
✓ _Plan_ ➜ ✓ _Acquire_ ➜ ✓ _Prepare_ ➜ ✓ _Explore_ ➜ ✓ _Model_ ➜ 🟢 **Deliver**
- [ ] Prepare Jupyter Notebook of project details through data science pipeline
- [ ] With additional time, continue with the exploration beyond MVP
- [ ] Proof read and complete README and project repository
- [ ] Prepare slide deck presentation of project

## V. Modules

- [`acquire`](https://raw.githubusercontent.com/ray-zapata/project_regression_zillow/main/acquire.py)
- [`prepare`](https://raw.githubusercontent.com/ray-zapata/project_regression_zillow/main/prepare.py)
- [`explore`](https://raw.githubusercontent.com/ray-zapata/project_regression_zillow/main/explore.py)
- [`model`  ](https://raw.githubusercontent.com/ray-zapata/project_regression_zillow/main/model.py)

## VI. Project Reproduction

WIP

[[Return to Top]](#predicting-home-value-with-zillow)
