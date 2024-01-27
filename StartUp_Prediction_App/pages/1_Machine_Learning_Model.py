import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from joblib import load

# Load the trained model and transformers
FILE_DIR1 = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(FILE_DIR1, os.pardir)
dir_of_interest = os.path.join(FILE_DIR, "resourses")
DATA_PATH = os.path.join(dir_of_interest, "Data")
MODEL_PATH=os.path.join(dir_of_interest, "Model")
DATA_PATH1 = os.path.join(DATA_PATH, "clean_data.csv")
MODEL_PATH1=os.path.join(MODEL_PATH, "pipeline_rf.pkl")
df = pd.read_csv(DATA_PATH1)
#pipeline_rf=load(MODEL_PATH1)
#st.dataframe(df)
df.rename({'status':'isClosed'}, axis=1, inplace=True)
df['isClosed'].replace({'operating':0,'ipo':0,'acquired':1,'closed':1}, inplace=True)

# Function to preprocess user input
def preprocess_input(data):
    # Extract numeric and categorical columns
    numeric_columns = ['founded_at', 'last_funding_at', 'funding_rounds',
                       'funding_total_usd', 'last_milestone_at', 'milestones',
                       'relationships', 'lat', 'lng', 'active_days']
    
    categorical_columns = ['country_code', 'category_code']
    
    # Extract and preprocess numeric features
    numeric_features = data[numeric_columns]
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features)
    
    # Extract and preprocess categorical features
    categorical_features = data[categorical_columns]
    encoder = LabelEncoder()
    
    # Initialize an empty DataFrame for the label encoded categorical features
    label_encoded_categorical_features = pd.DataFrame()
    
    # Apply LabelEncoder to each categorical column
    for col in categorical_features:
        values = encoder.fit_transform(data[col])
        label_encoded_categorical_features[col] = values

    # Combine the processed numeric and categorical features
    processed_data = np.concatenate([numeric_features_scaled, label_encoded_categorical_features.to_numpy()], axis=1)
    
    return processed_data


#Train random forest
X=preprocess_input(df)
y=df['isClosed']
from imblearn.over_sampling import SMOTE
smote=SMOTE()
X,y=smote.fit_resample(X,y)
rf=RandomForestClassifier()
rf.fit(X,y)

# Create a Streamlit app
st.title("Predict Company Status")

# Get user input
founded_at = st.slider("Select Founded Year", min_value=min(df['founded_at']), max_value=2024)
last_funding_at = st.slider("Select Last Funding Year", min_value=min(df['last_funding_at']), max_value=2024)
last_milestone_at = st.slider("Select Last Milestone Year", min_value=min(df['last_milestone_at']), max_value=2024)
funding_rounds = st.number_input("Select Funding Rounds")
funding_total_usd = st.number_input("Select Funding Totals in USD")
relationships = st.number_input("Select Relationships")
milestones = st.number_input("Select Milestones")
lat = st.number_input("Select Latitude")
lng = st.number_input("Select Longitude")
active_days = st.number_input("Select Active Days")

country_code = st.selectbox("Enter Country Code (e.g., 'US'):", df['country_code'].unique())
category_code = st.selectbox("Enter Category Code (e.g., 'TECH'):", df['category_code'].unique())
# Add more text_input fields for other categorical features

# Create a DataFrame with user input
user_input = pd.DataFrame({
    'founded_at': [founded_at],
    'last_funding_at': [last_funding_at],
    'last_milestone_at':[last_milestone_at],
    'funding_rounds':[funding_rounds],
    'country_code': [country_code],
    'category_code': [category_code],
    'funding_total_usd':[funding_total_usd],
    'relationships':[relationships],
    'milestones':[milestones],
    'lat':[lat],
    'lng':[lng],
    'active_days':[active_days]
})

# Preprocess user input
user_input_processed = preprocess_input(user_input)

# Make prediction
prediction = rf.predict(user_input_processed)

# Display prediction
st.subheader("Prediction:")
if prediction[0] == 0:
    st.write("The company is predicted to be open.")
else:
    st.write("The company is predicted to be closed.")
