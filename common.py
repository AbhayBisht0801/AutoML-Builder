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
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
import time
from langchain.agents import AgentExecutor, create_tool_calling_agent
load_dotenv()
llm=ChatGoogleGenerativeAI(model='gemini-1.0-pro')
llm1=ChatGoogleGenerativeAI(model='gemini-1.5-pro')
# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
def data_insights(path):
    if path.endswith('csv')==True:
        df=pd.read_csv(path)
    else:
        df=pd.read_excel(path)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    numeric_columns = list(df.select_dtypes(include=numerics).columns)
    categorical_columns = [i for i in df.columns if i not in numeric_columns]
    missing_value=(df.isnull().sum()*100)/df.shape[0]
    return categorical_columns,numeric_columns,missing_value.to_dict()
def preprocessing(categorical_feature, numerical_columns, missing_value_columns, user_requirement, llm1):
    prompt = '''
   Given the {user_requirement}, perform the necessary preprocessing steps based on the csv dataset provided, which includes:

Numeric Columns: {numerical_columns}
Categorical Columns: {categorical_feature}
Columns with Missing Values: {missing_value_columns}
based on the above info about dataset follow the instructions below for data preprocessing:

- Handle Missing Values: If the dataset contains missing values, address them according to the context of the problem statement. In real-time data, it is not always ideal to impute missing values, so use an appropriate method or leave them if justified. If no missing values exist, proceed to the next steps.

- Remove Irrelevant Columns: Identify and drop columns  that do not contribute to model performance or irrelevant columns for  example like customerId,name with respect to customer data.Identify the irrelevant columns from {numerical_columns}
and {categorical_feature}.

- Data Cleaning: Perform any additional cleaning steps such as removing duplicates, handling outliers, or correcting data formats as required.

- Encode Categorical Columns: Use suitable encoding techniques (e.g., One-Hot Encoding or Label Encoding) to convert categorical variables into numeric representations.

- Scale Numeric Columns: Apply appropriate scaling methods (e.g., StandardScaler, MinMaxScaler) to normalize the numeric features for optimal model performance.
- Train_Test_split: Split the Data into Train Test  and save it into a 'project/artifact' folder
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
    
def parse_output(response):
    parser=PydanticOutputParser(pydantic_object=CodeOutput)
    print(parser.get_format_instructions())
    code_parser_template= """Parse the response from a previous LLM into a description and a string of valid code, 
                                also come up with a valid filename this could be saved as that doesnt contain special characters. 
                                Here is the response: {response}. You should parse this in the following JSON Format: """
    json_prompt_tmpl=PromptTemplate(template=code_parser_template,input_variables=['response'],output_parser=parser)
    output=llm1.invoke(json_prompt_tmpl.format(response=response))
    result=output.content
    print(result)
    output=result.replace('```json','').replace('```','')
    return ast.literal_eval(output)
def code_to_function(code):
    template='''Convert the given code {code} into  function  with saving preprocessing model in pickle with their respective name so
      that it can be used during model prediction.put the explantion string in #  comments'''
    prompt=PromptTemplate(template=template,input_variables=['code'])
    return llm.invoke(prompt.format(code=code)).content
def code_to_file(code,file_name):
    with open(file_name, "w") as f:
        f.write(code.replace('```python','').replace('```',''))
        f.close()

def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
def preprocessing(categorical_feature, numerical_columns, missing_value_columns, user_requirement, llm1):
    prompt = '''
   Given the {user_requirement}, perform the necessary preprocessing steps based on the csv dataset provided, which includes:

Numeric Columns: {numerical_columns}
Categorical Columns: {categorical_feature}
Columns with Missing Values: {missing_value_columns}
based on the above info about dataset follow the instructions below for data preprocessing:

- Handle Missing Values: If the dataset contains missing values, address them according to the context of the problem statement. In real-time data, it is not always ideal to impute missing values, so use an appropriate method or leave them if justified. If no missing values exist, proceed to the next steps.

- Remove Irrelevant Columns: Identify and drop columns  that do not contribute to model performance or irrelevant columns for  example like customerId,name with respect to customer data.Identify the irrelevant columns from {numerical_columns}
and {categorical_feature}.

- Data Cleaning: Perform any additional cleaning steps such as removing duplicates, handling outliers, or correcting data formats as required.

- Encode Categorical Columns: Use suitable encoding techniques (e.g., One-Hot Encoding or Label Encoding) to convert categorical variables into numeric representations.

- Scale Numeric Columns: Apply appropriate scaling methods (e.g., StandardScaler, MinMaxScaler) to normalize the numeric features for optimal model performance.
- Train_Test_split: Split the Data into Train Test  and save it into a folder
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







