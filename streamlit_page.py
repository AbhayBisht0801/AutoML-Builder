import streamlit as st
import pandas as pd
st.header('Upload the respective dataset')
problem_statement=st.text_input()
uploaded_file=st.file_uploader('Upload the respective csv file',type=['csv','xlsx'])
if st.button('Submit'):
    pass
st.header('Upload the respective dataset')
uploaded_file=st.file_uploader('Upload the respective csv file',type=['csv','xlsx'])
if st.button('Submit'):
    pass
