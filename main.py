from common import data_insights,preprocessing,parse_output,code_to_function,llm1,code_to_file,create_directories
import pandas as pd
import time
create_directories(['project'])
categorical_feature,numerical_column,missing_value_column=data_insights(r"C:\\Users\bisht\Downloads\\titanic_toy.csv")
print(categorical_feature,numerical_column)
loop=0
while loop<2:
    output=preprocessing(categorical_feature=categorical_feature,numerical_columns=numerical_column,missing_value_columns=missing_value_column
                ,user_requirement='Create a classication model for prediction the people who survived in the titanic',llm1=llm1)
    time.sleep(2)
    loop+=1
    
code=parse_output(output)
code_func=code_to_function(code=code['code'])
code_to_file(code=code_func,file_name='preprocessing.py')










