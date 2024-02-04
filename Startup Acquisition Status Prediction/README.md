# Project Title: Start Up Acquisition Status Prediction

## Table of Content
1. Objective of the Project
2. Data info
3. Loading the packages or libraries
4. Loading the dataset
5. Understandng our data
6. Data Cleaning
     i. Delete irrelevant & redundant information
     ii. Remove noise or unreliable data (duplicate, missing values and outliers).
     iii. Data Transformation
     iv. Feature Creation
7. Exploratory Data Analysis
     i. Univariate Analysis
     ii. Bivariate Analysis
     iii. Multivariate Analysis
8. Feature Engineering
     i. Feature Selection Techniques
     ii. Feature Reduction Techniques
10. ML Model Building
11. Hyperparameter Tuning to select best model
12. Select Best Model
13. Build Pipeline
14. Create STreamlit Application

## 1. Objective of the Project
- The objective of the project is to predict whether a startup which is currently Operating, IPO, Acquired, or closed.This problem will be solved through a Supervised Machine Learning approach by training a model based on the history of startups which were either acquired or closed.

## 2. Data Info
- This Dataset of Crunchbase companies data.
- There are 196562 rows and 44 columns out of which will be used as features. The rest provide more information about the data, but will not be used for model training (like normalized name, entity id, short description etc.)
        - `entity_type` : Type of company.
        - `entity_id` : Unique Id for each entity.
        - `name` : Name of the entity or company.
4. `category_code` : Type of company.
5. `status` : Status of the company, whether it is operating or not and this is our target variable.
6. `founded_at` : Company foundation yera.
7. `closed_at` : Company shut-down year.
8. `country_code, state_code, city, region` : Country, State, City and region which the company located at.
9. `first_investment_at, last_investment_at` : First and Last Investment Dates.
10. `investment_rounds` : Count of Investment rounds take place.
11. `invested_companies` : Count of companies who are invested in this company.
12. `first_funding_at, last_funding_at` : First and Last funding date.
13. `funding_rounds` : Count of funding rounds.
14. `funding_total_usd` : Amount of total funding in US Dollar.
15. `first_milestone_at, last_milestone_at` : Date on which company achieved their first and last targeted milestone.
16. `milestones` : Number of milestones achieved by each company.
17. `relationships` : Number of relations company has with various stackholders.
18. `created_by` : Company Creator name.
19. `created_at, updated_at` : Company Created and Updated dates.
20. `lat, lng` : Latitude and Longitude of the company location.
21. `ROI` : Return of Investment.

## 3. Loading the packages or libraries
- Importing `pandas` and `numpy` for manipulaing data.
- Importing `matplotlib, seaborn, plotly` for data visualization.
- importing `scikitlearn` to building models and preprocessing data.

## 4. Loading Data Set
- There are 196562 rows and 44 columns out of which will be used as features. The rest provide more information about the data, but will not be used for model training (like normalized name, entity id, short description etc.)

## 5. Understanding Our Data
- `Shape` and `size` of dataset.
- Check for `columnns` in dataset.
- Check `Datatypes` of all columns.
- Check `statistical information`.
- Confirm whether data contain any `NULL` and `Duplicate` values.

## 6. Data Cleaning
### i. Delete irrelevant & redundant information
- Delete `region, city, state_code` as they provide too much of granularity.
- Delete `id, Unnamed: 0.1, entity_type, entity_id, parent_id, created_by, created_at, updated_at` as they are redundant.
- Delete `name`, domain`, `homepage_url`, `twitter_username`, `logo_url`, `logo_width`, `logo_height`, `short_description`, `description`, `overview`,`tag_list`, `name`, `normalized_name`, `permalink` as they are irrelevant features.

### ii. Remove noise or unreliable data (duplicate, missing values and outliers).
- Drop the columns which contains more than 97% of null rows
- Delete instances with missing values for `country_code`, `category_code`, `founded_at` `first_funding_at`, `first_milestone_at`, `relationships` and `lat`. (Since these are the type of data where adding value via imputation will create wrong pattern only.)
- Fill the missing values in numerical columns, `funding_total_usd` by median. (The median is less sensitive to extreme values compared to the mean. Imputing missing values with the median can be a good choice for right-skewed data as it is robust to outliers.)
- All the 5 columns contains Outliers. But the exstreme outliers found in only three columns, `relationships`, `active_days and `funding_total_usd`. Used IQR method to clip these outliers.

### iii. Data Transformation
- Convert data type of Date column from object to datetime and the extract only year.
- Convert data type of `relationships`, `funding_rounds`, `milestones` from float to integer.
- Generalize `category_code`- Since there are 42 categories, one-hot encoding which is going to create a lot of columns so Lets Check the repetition of value in ascending order and keep the first 15 values and name remaining one as other.
- Generalize `country_code`. Since there are 95 counties, one-hot encoding which is going to create a lot of columns so Lets Check the repetition of value in ascending order and keep the first 15 values and name remaining one as other.

### iv. Feature Creation
- Creating `active_days` feature as below,
     - Replacing values in closed_at columns using below condition:
     - if the value in status is `operating` then in closed_at, Let`s put 2021.
     - Where as if the value is `not-operating`, let`s put 0.
     - Subtract founded_date from closed_date, and calculate age in days.
     - Then drop the closed_at column.

## 
