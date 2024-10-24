import streamlit as st
import pandas as pd
st.title('AutoML')
st.subheader('Project Statement')
problem_statement=st.text_input('')
st.subheader('Upload the respective dataset')
uploaded_file=st.file_uploader('Upload the respective csv file',type=['csv','xlsx'])
st.subheader('Project Path (else same project directory )')
project_path=st.text_input('project Path')  

if st.button('Submit'):
    pass

