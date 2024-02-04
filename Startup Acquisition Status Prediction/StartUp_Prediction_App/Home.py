import streamlit as st
from matplotlib import image
import os

#Title of the home page
st.title("Machine Learning Model to Predict Acquisition Status")
#Using subheader
st.subheader('By:Amarendu, Fatima, Gangothri, Hiba, Suraj')

#Adding Image
FILE_DIR1 = os.path.dirname(os.path.abspath(__file__))
#FILE_DIR = os.path.path(os.pardir)
dir_of_interest = os.path.join(FILE_DIR1, "resourses")
IMAGE_PATH = os.path.join(dir_of_interest, "Image")
IMAGE_PATH1 = os.path.join(IMAGE_PATH, "pic.jpg")

img = image.imread(IMAGE_PATH1)
st.image(img)

#Using markdown cell type
st.markdown("Connect with me on,")

#Creating column layout
col1,col2,col3,col4=st.columns(4, gap='small')
with col1:
    st.write("[LinkedIn](https://www.linkedin.com/in/surajhonkamble)")
with col2:
    st.write("[GitHub](https://github.com/suraj_08051996)")
with col3:
    st.write("[Portfolio](https://linktr.ee/surajhonkamble)")
with col4:
    st.write("[Tableau](https://public.tableau.com/app/profile/suraj.honkamble)")
