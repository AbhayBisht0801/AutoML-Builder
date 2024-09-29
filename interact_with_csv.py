import os
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from load_dotenv import load_dotenv
from pydantic import BaseModel,Field
import pandas as pd
from pandasai import Agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
import ast
load_dotenv()
llm=ChatGoogleGenerativeAI(model='gemini-pro')
llm1=ChatGoogleGenerativeAI(model='gemini-1.5-pro')
# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
def data_insights(path):
    agent = create_csv_agent(llm=llm, path=path, allow_dangerous_code=True, verbose=True)

    # Identify categorical columns and map them to their unique values
    categorical_feature = agent.invoke('Identify the categorical columns in the CSV file and create a dictionary that maps each categorical column to its unique values.')
    print(categorical_feature)

    # Extract the names of the numerical columns
    numerical_column_names = agent.invoke('List all numerical columns in the dataset.')['output']
    print(numerical_column_names)

    # Find columns with missing values and calculate the percentage of missing data for each
    missing_value_column = agent.invoke('Identify columns with missing values and calculate the percentage of missing data for each column. Return this as a dictionary with column names as keys and missing percentages as values.')['output']
    print(missing_value_column)

    # Determine irrelevant columns that should not be used for machine learning model training
    irrelevant_columns = agent.invoke('Identify columns that are irrelevant and should not be used for training a machine learning model.')['output']
    print(irrelevant_columns)

    return categorical_feature, numerical_column_names, missing_value_column, irrelevant_columns

def preprocessing(categorical_feature, numerical_columns, missing_value_columns, user_requirement, llm1):
    prompt = '''
   Given the user requirements, perform the necessary preprocessing steps based on the dataset provided, which includes:

Numeric Columns: {numerical_columns}
Categorical Columns: {categorical_feature}
Columns with Missing Values: {missing_value_columns}
Follow the instructions below for data preprocessing:

Handle Missing Values: If the dataset contains missing values, address them according to the context of the problem statement. In real-time data, it is not always ideal to impute missing values, so use an appropriate method or leave them if justified. If no missing values exist, proceed to the next steps.

Remove Irrelevant Columns: Drop columns  that do not contribute to model performance.

Data Cleaning: Perform any additional cleaning steps such as removing duplicates, handling outliers, or correcting data formats as required.

Encode Categorical Columns: Use suitable encoding techniques (e.g., One-Hot Encoding or Label Encoding) to convert categorical variables into numeric representations.

Scale Numeric Columns: Apply appropriate scaling methods (e.g., StandardScaler, MinMaxScaler) to normalize the numeric features for optimal model performance.
keep the heading in # .
    '''
    
    # Create the template
    template = PromptTemplate(
        template=prompt, 
        input_variables=['categorical_feature', 'numerical_columns', 'missing_value_columns', 'user_requirement']
    )
    
    # Format the prompt
    formatted = template.format(
        categorical_feature=categorical_feature, 
        numerical_columns=numerical_columns, 
        missing_value_columns=missing_value_columns, 
        user_requirement=user_requirement
    )
    
    # Call the LLM to get the code based on the formatted prompt
    output=llm1.invoke(formatted).content
    return output
    
class CodeOutput(BaseModel):
    code: str =Field(description='Code of the preprocessing')
    description: str =Field(description='Explanation of Preprocessing Code')
    filename: str=Field(description='Give the respective filename')
    
def Code_to_py(response):
    parser=PydanticOutputParser(pydantic_object=CodeOutput)
    print(parser.get_format_instructions())
    code_parser_template= """Parse the response from a previous LLM into a description and a string of valid code, 
                                also come up with a valid filename this could be saved as that doesnt contain special characters. 
                                Here is the response: {response}. You should parse this in the following JSON Format: """
    json_prompt_tmpl=PromptTemplate(template=code_parser_template,input_variables=['response'],output_parser=parser)
    output=llm1.invoke(json_prompt_tmpl.format(response=response))
    result=output.content.replace("```json", "").replace("```", "")
    return eval(result)
def code_to_function(code):
    template='Convert the given code {code} into  function with saving preprocessing model in pickle with their respective name so that it can be used during model prediction'
    prompt=PromptTemplate(template=template,input_variables=['code'])
    return llm.invoke(prompt.format(code=code)).content
response="""Let's break down how to preprocess your Titanic dataset given the provided information.

**1. Understanding the Data**

* **Numeric Columns:** 'Age', 'Fare', 'Family', 'Survived'
* **Categorical Columns:** You'll need to identify these from the CSV. Common ones might be 'Sex', 'Pclass' (passenger class), 'Embarked' (port of embarkation), 'Cabin' (cabin number), and 'Ticket' (ticket number).
* **Missing Values:** You have some missing values in 'Age' and 'Fare'.

**2. Preprocessing Steps**

Here's a Python-based preprocessing approach using the Pandas and Scikit-learn libraries:

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# **1. Load the Data**
data = pd.read_csv("your_titanic_dataset.csv")

# **2. Handle Missing Values**

# **Age:** Since age is likely an important predictor, impute missing ages.
# You could use the median age of passengers in the same Pclass.
data['Age'] = data.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

# **Fare:** Missing fares are less common. You can impute with the median fare.
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# **3. Remove Irrelevant Columns**

# Consider removing columns like:
# * 'PassengerId':  Just an identifier.
# * 'Name':  Unlikely to be directly useful (though you could extract titles like 'Mr.', 'Mrs.' for potential feature engineering).
# * 'Ticket':  Likely too complex and varied to be useful directly.
# * 'Cabin':  Has many missing values and might not be a strong predictor after feature engineering.

data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# **4. Data Cleaning**

# **Duplicates:** Check for and remove duplicate rows if any.
data.drop_duplicates(inplace=True)

# **Outliers:** You might analyze 'Age' and 'Fare' for outliers. Consider capping extreme values instead of removal to preserve information.       

# **5. Encode Categorical Columns**

# **Identify Categorical Columns:**
categorical_cols = ['Sex', 'Embarked', 'Pclass'] # Replace with your actual columns

# **One-Hot Encoding:**
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # Use sparse=True for large datasets
encoded_cols = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
data = data.drop(categorical_cols, axis=1).reset_index(drop=True)
data = pd.concat([data, encoded_df], axis=1)

# **6. Scale Numeric Columns**

# **Select Numeric Columns:**
numeric_cols = ['Age', 'Fare', 'Family']

# **Standard Scaling:**
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# **Your preprocessed data is now ready for modeling!**
print(data.head())
```"""
output=Code_to_py(response=response)

print(output)

print(code_to_function(output['code']))




