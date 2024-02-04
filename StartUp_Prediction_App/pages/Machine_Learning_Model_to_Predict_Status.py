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
MODEL_PATH1 = os.path.join(MODEL_PATH, "RF_for_bivariate.pkl")
MODEL_PATH2=os.path.join(MODEL_PATH, "GB_for_mulivariate.pkl")

# Page Title
st.write("""
# Acquisition Status App
This app predicts the **acquisition status** of a startup company!
""")

#Taking user input
st.sidebar.header('User Input Parameters')

st.sidebar.markdown("""
""")

# Taking user input as a csv file from local browser or Manual input
file = st.sidebar.file_uploader("Upload your input file in CSV format", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
else:
    def user_input_features():
        founded_at = st.sidebar.slider('founded_at', 1911, 2022, 2000)
        #investment_rounds = st.sidebar.slider('investment_rounds', 1.0, 10.0, 1.0)
        funding_rounds = st.sidebar.slider('funding_rounds', 1.0, 3.0, 1.0)
        funding_total_usd = st.sidebar.slider('funding_total_usd', 100, 20000000, 10000)
        milestones = st.sidebar.slider('milestones',  1.0, 5.0, 1.0)
        relationships = st.sidebar.slider('relationships', 1.0, 15.0, 1.0)
        #roi = st.sidebar.slider('ROI', 0.1, 1000.0, 0.2)
        active_days = st.sidebar.slider('active_days', 100, 10000, 3000)

        category_code = st.sidebar.selectbox('Category Code',('advertising','analytics','biotech','cleantech',
                                                    'ecommerce','education','enterprise','finance','social','network_hosting',
                                                    'games_video', 'hardware','mobile','other' ,'software', 'web'))
        country_code = st.sidebar.selectbox('Country Code',('AUS','CAN','DEU', 'ESP',
                                                    'FRA', 'GBR', 'IND','ISR',
                                                    'USA','other'))
        first_milestone_at = st.sidebar.slider('first milestone at', 1976, 2022, 2000)
        last_milestone_at = st.sidebar.slider('last milestone at', 1976, 2022, 2000)
        first_funding_at = st.sidebar.slider('first_funding_at', 1960, 2022, 2000)
        last_funding_at = st.sidebar.slider('last_funding_at', 1960, 2022, 2000)
        lat= st.sidebar.slider("Latitude",-42.883611,70.919200, 70.919200)
        lng= st.sidebar.slider("Longitude",-158.056896,174.776236, 174.776236)
        data = {'founded_at': founded_at,
                #'investment_rounds': investment_rounds,
                'funding_rounds': funding_rounds,
                'funding_total_usd': funding_total_usd,
                'milestones': milestones,
                'relationships': relationships,
                #'ROI': roi,
                'lat':lat,
                'lng':lng,
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

country_code = pd.get_dummies(df['country_code'], prefix="country_code")
category_code = pd.get_dummies(df['category_code'], prefix="category_code")

df.drop(['country_code', 'category_code'], axis=1, inplace=True)

# Concatenate the one-hot-encoded columns with the original DataFrame
df = pd.concat([df, country_code, category_code], axis=1)

# Adjust the column names based on your actual model's feature names
expected_columns = ['founded_at', 'first_funding_at', 'last_funding_at',
       'funding_rounds', 'funding_total_usd', 'first_milestone_at',
       'last_milestone_at', 'milestones', 'relationships', 'lat', 'lng',
       'active_days', 'country_code_AUS', 'country_code_CAN',
       'country_code_DEU', 'country_code_ESP', 'country_code_FRA',
       'country_code_GBR', 'country_code_IND', 'country_code_ISR',
       'country_code_USA', 'country_code_other', 'category_code_advertising',
       'category_code_analytics', 'category_code_biotech',
       'category_code_cleantech', 'category_code_ecommerce',
       'category_code_education', 'category_code_enterprise',
       'category_code_finance', 'category_code_games_video',
       'category_code_hardware', 'category_code_mobile',
       'category_code_network_hosting', 'category_code_other',
       'category_code_social', 'category_code_software', 'category_code_web']

# Add any missing columns with value 0
for col in expected_columns:
    if col not in df.columns:
        df[col] = 0

# Ensure the order of columns is the same as the model expects
df = df[expected_columns]
st.subheader('User Input parameters')
st.write(df)


Reg_RF = pickle.load(open(MODEL_PATH1, 'rb'))

Operating_prob = Reg_RF.predict_proba(df)[:,1]
RF_training_result = pd.DataFrame()
RF_training_result["Operating_prob"] = Operating_prob
RF_training_result['predicted'] = RF_training_result.Operating_prob.map(lambda x: 1 if x > 0.8 else 0)
#st.write(Reg_RF.predict(df))

# Apply model to make predictions
RF_prediction = RF_training_result['predicted'].values
prediction_proba = Reg_RF.predict_proba(df)
prediction_proba = np.round(prediction_proba,2)

st.subheader('Expected Startup status')
classes_bin = np.array(["Operating","Closed"])
classes_mul = np.array(["Operating","IPO","Acquired","Closed"])
#st.write(classes)
    
if RF_prediction == 0:
    GB = pickle.load(open(MODEL_PATH2, 'rb'))
    GB_prediction = GB.predict(df)
    
    if st.button('Predict using Bivariate Model'):
        st.markdown("<h3 style='color: #9E6273 ; font-weight: 100;'>Binary Class Prediction</h3>", unsafe_allow_html=True)
        st.write(classes_bin[RF_prediction])
        st.markdown("<h3 style='color: #9E6273 ; font-weight: 100;'>Binary Class Probability</h3>", unsafe_allow_html=True)
        st.write(prediction_proba)
        
    if st.button('Predict using Multivariate Model'):
        st.markdown("<h3 style='color: #9E6273 ; font-weight: 100;'>Multi Class Prediction</h3>", unsafe_allow_html=True)
        st.write(classes_mul[GB_prediction])
        st.markdown("<h3 style='color: #9E6273 ; font-weight: 100;'>Multi Class Probability</h3>", unsafe_allow_html=True)
        GB_prediction_proba = GB.predict_proba(df)
        GB_prediction_proba = np.round(GB_prediction_proba,2)
        st.write(GB_prediction_proba)
    
    st.subheader('Final Prediction')
    st.write(classes_mul[GB_prediction])
else:
    if st.button('Predict using Binary Class Model'):
        st.subheader('Binary Class Prediction')
        st.write(classes_bin[RF_prediction])

        st.subheader('Binary Class Probability')
        st.write(prediction_proba)
    st.subheader('Final Prediction')
    st.write(classes_bin[RF_prediction])