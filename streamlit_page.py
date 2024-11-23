import streamlit as st
import pandas as pd
st.title('AutoML')
st.subheader('Project Statement')
problem_statement=st.text_input('')
st.subheader('Upload the respective dataset')
uploaded_file=st.file_uploader('Upload the respective csv file',type=['csv','xlsx']) 
if st.button('Submit'):
    with st.spinner('Please Wait'):
        df=pd.read_csv(uploaded_file)
        df.to_csv('data.csv')

