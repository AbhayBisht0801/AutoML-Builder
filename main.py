from interact_with_csv import data_insights,preprocessing,Code_to_py,llm1
import pandas as pd
categorical_feature,numerical_column,missing_value_column,irrelevant_column=data_insights(r"C:\\Users\bisht\Downloads\\titanic_toy.csv")

output=preprocessing(categorical_feature=categorical_feature,numerical_columns=numerical_column,missing_value_columns=missing_value_column
              ,user_requirement='Create a classication model for prediction the people who survived',llm1=llm1)

print(output)

result=Code_to_py(output)
print(result)