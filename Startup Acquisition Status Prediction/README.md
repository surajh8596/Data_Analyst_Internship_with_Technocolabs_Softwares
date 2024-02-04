# Project Title: Start Up Acquisition Status Prediction
# Table of Content
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

1. entity_type : Type of company.
2. entity_id : Unique Id for each entity.
3. name : Name of the entity or company.
4. category_code : Type of company.
5. status : Status of the company, whether it is operating or not and this is our target variable.
6. founded_at : Company foundation yera.
7. closed_at : Company shut-down year.
8. country_code, state_code, city, region : Country, State, City and region which the company located at.
9. first_investment_at, last_investment_at : First and Last Investment Dates.
10. investment_rounds : Count of Investment rounds take place.
11. invested_companies : Count of companies who are invested in this company.
12. first_funding_at, last_funding_at : First and Last funding date.
13. funding_rounds : Count of funding rounds.
14. funding_total_usd : Amount of total funding in US Dollar.
15. first_milestone_at, last_milestone_at : Date on which company achieved their first and last targeted milestone.
16. milestones : Number of milestones achieved by each company.
17. relationships : Number of relations company has with various stackholders.
18. created_by : Company Creator name.
19. created_at, updated_at : Company Created and Updated dates.
20. lat, lng : Latitude and Longitude of the company location.
21. ROI : Return of Investment.
