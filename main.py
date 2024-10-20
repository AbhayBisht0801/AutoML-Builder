from common import data_insights,preprocessing,Code_to_py,code_to_function,llm1,code_to_file
import pandas as pd
categorical_feature,numerical_column,missing_value_column=data_insights(r"C:\\Users\bisht\Downloads\\titanic_toy.csv")
print(categorical_feature,numerical_column)
output=preprocessing(categorical_feature=categorical_feature,numerical_columns=numerical_column,missing_value_columns=missing_value_column
              ,user_requirement='Create a classication model for prediction the people who survived in the titanic',llm1=llm1)

print(output)
result=Code_to_py(output)
func=code_to_function(result['code'])




