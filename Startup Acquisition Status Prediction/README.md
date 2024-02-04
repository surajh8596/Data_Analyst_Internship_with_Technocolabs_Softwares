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
9. ML Model Building, Hyperparameter Tuning
10. Select Best Model
11. Build Pipeline
12. Create STreamlit Application

## 1. Objective of the Project
- The objective of the project is to predict whether a startup which is currently Operating, IPO, Acquired, or closed.This problem will be solved through a Supervised Machine Learning approach by training a model based on the history of startups which were either acquired or closed.

## 2. Data Info
- This Dataset of Crunchbase companies data.
- There are 196562 rows and 44 columns out of which will be used as features. The rest provide more information about the data, but will not be used for model training (like normalized name, entity id, short description etc.)
1. `entity_type` : Type of company.
2. `entity_id` : Unique Id for each entity.
3. `name` : Name of the entity or company.
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
- Delete `name`, `domain`, `homepage_url`, `twitter_username`, `logo_url`, `logo_width`, `logo_height`, `short_description`, `description`, `overview`,`tag_list`, `name`, `normalized_name`, `permalink` as they are irrelevant features.

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

## 7. Exploratory Data Analysis
### i. Univariate Analysis
Below are the some business insights from `Univariate Analysis`,
- The most frequently occurring category is `Other,` representing 2,237 instances, indicating a diverse range of businesses. Following closely is the `Software` category with 1,590 companies, emphasizing a significant presence in the software sector.
- While the majority of companies, constituting 94.0%, are currently operating, it`s important to note a class imbalance, with only 6% categorized as not-operating.
- Trend is from earlist to newsest in all above date related columns. Most companies founded in 2011, most companies recieved their first funding in year 2012, maximum companies recieved last funding in the year 2013. Maximum companies reached their first milestone in 2012 and the last milestone is in the year 2013.
- Most of the entities or the copanies resides in USA, followed by Other country.

### ii. Bivariate Analysis
Below are the some business insights from `Bivariate Analysis`,
- Most of the entities or the copanies resides in USA, followed by Other country.
- The USA has the highest number of companies across almost all categories. Other countries, such as the United Kingdom (GBR) and Canada (CAN), also have a significant presence in various categories.
- `Software` and `E-commerce` are among the most popular categories across all countries, with the USA leading in these sectors. The `Other` category also shows notable activity, suggesting a diverse range of industries or startups.
- Some countries exhibit specialization in specific categories; for example, Israel (ISR) has a relatively higher concentration in `Enterprise` and `Security.`
- Categories like `Biotech,` `Cleantech,` and `Analytics` have a presence across different countries, indicating a focus on emerging and innovative technologies.
- The United States (USA) stands out with the highest average number of funding rounds per company (approximately 1.56). Emerging market India (IND) shows least average funding rounds per company of approximately 1.27.
- The United States (USA) is the top recipient of funding, with a total of $4.64 million USD. Australia (AUS) and Spain (ESP) have least average funding totals.
- Companies in France (FRA) and the United Kingdom (GBR) demonstrate longer active days, with values of 4252.80 and 4181.84, respectively. This suggests a potentially longer operational longevity.
- The Analytics compananies stands out with the highest average number of funding rounds per company (approximately 1.7). Web category companies shows least average funding rounds per company of approximately 1.4.
- `BioTech` followed by `DeanTech` companies `operating most days` and Least operating companies belongs to `Social` category. Maximum average funding recieved by BioTech` followed by `DeanTech` companies and Least average funding is by `Social` companies.

### iii. Multivariate Analysis
Below are the some business insights from `multivariate Analysis`,
- The stronger negative correlations (closer to -1) with `isClosed` are observed for `active_days, last_funding_at, first_funding_at`, and `founded_at`.
- The weaker negative correlations (closer to 0) include `milestones, relationships, funding_rounds, funding_total_usd`.
- The positive correlations (although weak) are seen with `lat`, `lng`, and `milestones`.

## 8. Feature Engineering
We used 3 techniques,
### i. Correlation Analysis & Multico-linearity
Observation and feature selection based on above correlation Matrix:
- Multicolinearity exists between columns `first_funding_at` and `last_funding_at` with correlation coefficient `0.85`. also between `first_milestone_at` and `last_milestone_at` with coefficient value `0.86`. We will drop any one column.
- The stronger negative correlations (closer to -1) with "isClosed" are observed for `active_days`, `last_funding_at`, `first_funding_at`, `first_milestone_at`, `last_milestone_at` and `founded_at`.
- The weaker negative correlations (closer to 0) include `lng`, `relationships`, `funding_rounds`, `funding_total_usd`.
- The weaker positive correlations (closer to 0) are seen with `lat` and `milestones`.
- From above correlation matrix we observed that there are 5 features which have correlation coeffienct less than `+0.05` or `-0.05`. These features are `lng`, `lat`, `funding_rounds`, `funding_total_usd` and `milestones`. We can eliminate these features.

### ii. PCA Technique
Observation and feature selection based on PCA:
- 19 Principal Components are selected as Input Features.
- `X` contains the data represented along the selected principal components. These components capture the most significant information in the dataset while reducing dimensionality.

### iii. Mutual Information Technique
Observation and feature selection based on Mutual Information Scores:
- There are total 38 Independed features. Out of 36, 12 features have MI Score 0, Hence these features have no dependency or doest carry any information which will help us to predict target class.
- `founded_at`,  `last_funding_at`, `funding_rounds`, `funding_total_usd`, `last_milestone_at`, `relationships`, `milestones`, `lat`, `active_days`, `country_code_CAT`, `country_code_DEU`, `country_code_IND`,`country_code_ISR`, `country_code_USA`, `category_code_analytics`, `category_code_biotech`, `category_code_cleantech`, `category_code_education`, `category_code_enterprise`,`category_code_game_video`, `category_code_mobile`, `category_code_others`, `category_code_software` and `category_code_web`. These are the selected features having MI Score greater than Zero.

## 9. Machine Learning Model Building: Binary and Multi class models
Lets use top 10 classification models, traian then and test on testing dataset and evaluate them using `Accuracy score, precision, recall` because our data is Imbalanced.
1. Logistic Regression
2. Naive Bayes
3. KNN - K-Nearest Neighbors
4. SVC - Support Vector Classifier
5. Decision Tree Classifier
6. Random Forest Classifier
7. Extra Trees Classifier
8. Ada Boost Classifier
9. Gradient Boosting Classifier
10. XG Boost Classifier

## 10. Select Best models
- `Logistic Regression`, `Support Vector Classifier`, `Random Forest Classifier`, `Extra Trees Classifier` and `Gradient Boosting Classifier` exhibit remarkable performance with an accuracy of 100% on both training and testing datasets.
- These models demonstrate exceptional generalization as they have successfully classified all test values without any misclassifications.
- Although we did not observed `Any Overfitting` but still to Overcome `Overfitting` we have used `Feature Selection`, `Cross Validation`, `Hyperparamer Tunning`, `Regularization Techniques` in Logistic Regression and used `Ensemble Models` like Random Forest, Extra Trees, Gradient Boosting, Ada Boost and XgBoost.
- This outstanding performance suggests that these models are well-suited for the given task.
- Selecting `Random Forest Classifier` as a best model to predict `BINARY` target class. `Gradient Boosting Classifier` model as best model to predict `MULTICLASS` target.

## 11. Build Binary and Multiclass Pipeline
- Selecting best `Binary` class model based on higher precision of Operating class. We selected `Random Forest` which has `100% accuracy` on train and `97% accuracy` on test to build binary class pipeline.
- Building multivariate model based on the records that bivariate model was not able to predict as Operating class
- Selecting `Gradient Boosting` model based on higher accuracy to build multiclass pipeline.
- Serializing these two pipelines 

## 12. Creating Streamlit Application
- Used the Streamlit open-source library in Python to build a startup acquisition status prediction app. In this app, you will utilize serialized models within a pipeline. The binary pipeline consists of a Random Forest model as the primary model, while the multiclass pipeline incorporates a Gradient Boosting model as the primary classifier.
- The app will offer users the flexibility to input data either by uploading a CSV file or manually entering values for specific columns. Upon receiving user input, the data undergoes preprocessing steps to ensure it aligns with the model's requirements. Subsequently, the app makes predictions based on the input data.
- If the initial prediction from the binary pipeline is equal to 0, the app will then engage the multiclass pipeline for further prediction. This two-step process enhances the prediction accuracy and accommodates both binary and multiclass scenarios.
- Here is a high-level outline of the steps involved in building the app:
     - Run the Streamlit app, either locally or deploy it to a platform like Streamlit Sharing, Heroku, or another hosting service.
     - This app structure ensures a seamless user experience, allowing for both file-based and manual data input while incorporating the strengths of both binary and multiclass    predictive models for accurate startup acquisition status predictions.

App link - http://tinyurl.com/2s4tnm5f
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
