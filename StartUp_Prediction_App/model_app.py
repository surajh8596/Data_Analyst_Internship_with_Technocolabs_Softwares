import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle
import xgboost
import os

# Load the trained models
FILE_DIR1 = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(FILE_DIR1, os.pardir)
dir_of_interest = os.path.join(FILE_DIR, "resourses")
MODEL_PATH=os.path.join(dir_of_interest, "Model")
DATA_PATH1 = os.path.join(MODEL_PATH, "Regularised_LR_for_bivariate.csv")
MODEL_PATH2=os.path.join(MODEL_PATH, "xgboost_for_mulivariate.pkl")

# Define a custom style using Markdown
custom_style = """
    <style>
        .custom-subheader {
            font-size: 24px; 
            color: #D8708D; 
            font-weight: bold; 
            font-family: 'Arial', sans-serif; 
        }
    </style>
"""

# Apply the custom style
st.markdown(custom_style, unsafe_allow_html=True)


st.write("""
# Acquisition Status App

This app predicts the **acquisition status** of a startup company!
""")

st.sidebar.header('User Input Parameters')

st.sidebar.markdown("""
[)
""")

file = st.sidebar.file_uploader("Upload your input file in CSV format", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
else:
    def user_input_features():
        founded_at = st.sidebar.slider('founded_at', 1990, 2022, 1990)
        investment_rounds = st.sidebar.slider('investment_rounds', 1.0, 5.0, 1.0)
        funding_rounds = st.sidebar.slider('funding_rounds', 1.0, 5.0, 1.0)
        funding_total_usd = st.sidebar.slider('funding_total_usd', 10000, 2000000, 10000)
        milestones = st.sidebar.slider('milestones',  1.0, 10.0, 1.0)
        relationships = st.sidebar.slider('relationships', 1.0, 10.0, 1.0)
        roi = st.sidebar.slider('ROI', 0.1, 1000.0, 0.2)
        active_days = st.sidebar.slider('active_days', 2500, 10000, 3000)

        category_code = st.sidebar.selectbox('Category Code',('biotech','ecommerce', 'enterprise',
                                                    'games_video', 'hardware', 'health',
                                                    'mobile','other' ,'software', 'web'))
        country_code = st.sidebar.selectbox('Country Code',('CAN','DEU', 'ESP',
                                                    'FRA', 'GBR', 'IND', 'IRL','ISR',
                                                    'USA','other'))
        first_milestone_at = st.sidebar.slider('first milestone at', 1990, 2022, 2000)
        last_milestone_at = st.sidebar.slider('last milestone at', 1990, 2022, 2000)
        first_funding_at = st.sidebar.slider('first_funding_at', 1990, 2022, 2000)
        last_funding_at = st.sidebar.slider('last_funding_at', 1990, 2022, 2000)

        data = {'founded_at': founded_at,
                'investment_rounds': investment_rounds,
                'funding_rounds': funding_rounds,
                'funding_total_usd': funding_total_usd,
                'milestones': milestones,
                'relationships': relationships,
                'ROI': roi,
                'active_days': active_days,
                'category_code': category_code,
                'country_code': country_code,
                'first_milestone_at': first_milestone_at,     
                'last_milestone_at': last_milestone_at,
                'first_funding_at': first_funding_at,
                'last_funding_at': last_funding_at,
                }

        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()

df["milestone_diff"] = df['last_milestone_at'] - df['first_milestone_at']
df["funding_year_diff"] = df['last_funding_at'] - df['first_funding_at']
df.drop(columns = ["first_milestone_at", "last_milestone_at","last_funding_at", "first_funding_at"], inplace = True)

df['funding_usd_for_1_round'] = df['funding_total_usd']/df['funding_rounds']

df["Age_group"] = np.nan
# df.loc[(1000 < df["active_days"]) & ( df["active_days"] <= 2500),"Age_group"] = "1000-2500"
df.loc[(2500 < df["active_days"]) & ( df["active_days"] <= 4000),"Age_group"] = "2500-4000"
df.loc[(4000 < df["active_days"]) & ( df["active_days"] <= 5500),"Age_group"] = "4000-5500"
df.loc[(5500 < df["active_days"]) & ( df["active_days"] <= 7000),"Age_group"] = "5500-7000"
df.loc[(7000 < df["active_days"]) & ( df["active_days"] <= 8500),"Age_group"] = "7000-10000"

Age_group = pd.get_dummies(df['Age_group'], prefix="Age_group")
country_code = pd.get_dummies(df['country_code'], prefix="country_code")
category_code = pd.get_dummies(df['category_code'], prefix="category_code")

df.drop(['Age_group', 'country_code', 'category_code'], axis=1, inplace=True)

# Concatenate the one-hot-encoded columns with the original DataFrame
df = pd.concat([df, Age_group, country_code, category_code], axis=1)

# Adjust the column names based on your actual model's feature names
expected_columns = ['founded_at', 'investment_rounds', 'funding_rounds',
       'funding_total_usd', 'milestones', 'relationships', 'ROI',
       'active_days', 'category_code_biotech', 'category_code_ecommerce',
       'category_code_enterprise', 'category_code_games_video',
       'category_code_hardware', 'category_code_health',
       'category_code_mobile', 'category_code_other', 'category_code_software',
       'category_code_web', 'country_code_CAN', 'country_code_DEU',
       'country_code_ESP', 'country_code_FRA', 'country_code_GBR',
       'country_code_IND', 'country_code_IRL', 'country_code_ISR',
       'country_code_USA', 'country_code_other', 'funding_usd_for_1_round',
       'milestone_diff', 'funding_year_diff', 'Age_group_2500-4000',
       'Age_group_4000-5500', 'Age_group_5500-7000', 'Age_group_7000-10000']

# Add any missing columns with value 0
for col in expected_columns:
    if col not in df.columns:
        df[col] = 0

# Ensure the order of columns is the same as the model expects
df = df[expected_columns]
st.subheader('User Input parameters')
st.write(df)

Reg_LR = pickle.load(open('Regularised_LR_for_bivariate.pkl', 'rb'))

Operating_prob = Reg_LR.predict_proba(df)[:,1]
LR_training_result = pd.DataFrame()
LR_training_result["Operating_prob"] = Operating_prob
LR_training_result['predicted'] = LR_training_result.Operating_prob.map( lambda x: 1 if x > 0.9 else 0)

# Apply model to make predictions
LR_prediction = LR_training_result['predicted'].values
prediction_proba = Reg_LR.predict_proba(df)
prediction_proba = np.round(prediction_proba,2)

st.subheader('Expected Startup status')
classes = np.array(['Closed','Operating',"IPO","Acquired"])
st.write(classes)
    
if LR_prediction == 0:
    Xgb = pickle.load(open('xgboost_for_mulivariate.pkl', 'rb'))
    Xgb_prediction = Xgb.predict(df)
    
    if st.button('Predict using Bivariate Model'):
        st.markdown("<h3 style='color: #9E6273 ; font-weight: 100;'>Bivariate Prediction</h3>", unsafe_allow_html=True)
        st.write(classes[LR_prediction])
        st.markdown("<h3 style='color: #9E6273 ; font-weight: 100;'>Bivariate Probability</h3>", unsafe_allow_html=True)
        st.write(prediction_proba)
        
    if st.button('Predict using Multivariate Model'):
        st.markdown("<h3 style='color: #9E6273 ; font-weight: 100;'>Multivariate Prediction</h3>", unsafe_allow_html=True)
        st.write(classes[Xgb_prediction])
        st.markdown("<h3 style='color: #9E6273 ; font-weight: 100;'>Multivariate Probability</h3>", unsafe_allow_html=True)
        xgb_prediction_proba = Xgb.predict_proba(df)
        xgb_prediction_proba = np.round(xgb_prediction_proba,2)
        st.write(xgb_prediction_proba)
    
    st.subheader('Final Prediction')
    st.write(classes[Xgb_prediction])
else:
    if st.button('Predict using Bivariate Model'):
        st.subheader('Bivariate Prediction')
        st.write(classes[LR_prediction])

        st.subheader('Bivariate Probability')
        st.write(prediction_proba)
    st.subheader('Final Prediction')
    st.write(classes[LR_prediction])