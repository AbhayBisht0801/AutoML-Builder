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
code=parse_output(result)
print(code)
code_func=code_to_function(code=code['code'])
code_to_file(code=code_func,file_name='preprocessing.py')
