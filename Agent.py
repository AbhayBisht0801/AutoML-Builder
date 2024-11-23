import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from load_dotenv import load_dotenv
load_dotenv()
from common import gen_to_py,run_generated_code,correct_code,create_directories,requirement,project_setup
import subprocess
import os
import logging
import time
import shutil
llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro')
os.makedirs('project/logs', exist_ok=True)
logging.basicConfig(filename='project/logs/code_correction.log', 
                            level=logging.INFO, 
                            format='%(asctime)s - %(message)s')

class MLAagent:
    def __init__(self,problem_statement,dataset_path,llm):
        self.problem_statement = problem_statement
        
        self.dataset_source=dataset_path
        
        self.llm=llm
        self.setup_directory()
        self.data_insights()
        self.preprocessing()
        self.Model_building()
        self.check_code()
    def setup_directory(self):
        logging.info('Intiallizing the setup')
        create_directories(['project'])
        data_dir=os.path.join('project','data')
        create_directories([data_dir])
        if self.dataset_source.endswith('csv')==True:
            shutil.copy(self.dataset_source,os.path.join(data_dir,'data.csv'))
        else:
            shutil.copy(self.dataset_source,os.path.join(data_dir,'data.xlsx'))
        self.dataset_path=os.path.join(data_dir,os.listdir(data_dir)[0])
        logging.info('Directory setup complete')
        
    def data_insights(self):
        logging.info(' Collecting Data Insights')
        
        if self.dataset_path.endswith('csv')==True:
            df=pd.read_csv(self.dataset_path)
        else:
            df=pd.read_excel(self.dataset_path)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        numeric_columns = list(df.select_dtypes(include=numerics).columns)
        categorical_columns = [i for i in df.columns if i not in numeric_columns]
        missing_value=(df.isnull().sum()*100)/df.shape[0]
        self.categorical_columns=categorical_columns
        self.numerical_columns=numeric_columns
        self.missing_value_columns=missing_value.to_dict()
    def preprocessing(self):
        logging.info('Starting the  Preprocessing of the data')
        prompt = """
Using the input details provided below, generate Python code for preprocessing the dataset:

Input Information:
- User Requirement: {user_requirement} (Describes the context and objective of the task).
- Dataset Path: {dataset_path} (Path to the dataset file
- Numerical Columns: {numerical_columns} (List of numerical features in the dataset).
- Categorical Columns: {categorical_columns} (List of categorical features in the dataset).
- Columns with Missing Values: {missing_value_columns} (Columns that contain missing values).

Instructions for Preprocessing:

1. Handle Missing Values:
   - If the dataset contains missing values, address them based on the context of the {user_requirement}. Use appropriate imputation methods, or leave them as is if justified.
   - If there are no missing values, skip this step.

2. Remove Irrelevant Columns:
   - Identify and drop columns that do not contribute to achieving the {user_requirement}.
   - For example, irrelevant columns could include unique identifiers like "customerId" or personal details like "name".
   - Irrelevant columns should be identified from {numerical_columns} and {categorical_columns}.

3. Data Cleaning:
   - Perform additional cleaning steps such as removing duplicate records, handling outliers, or correcting inconsistent data formats.

4. Encode Categorical Columns:
   - Convert {categorical_columns} into numeric representations using appropriate encoding techniques, such as One-Hot Encoding or Label Encoding.
   - Do not include any columns identified as irrelevant in this step.

5. Scale Numeric Columns:
   - Normalize {numerical_columns} using a suitable scaling method such as StandardScaler or MinMaxScaler to optimize model performance.

6. Split Dataset:
   - Split the dataset into training and testing subsets based on an appropriate ratio (e.g., 80:20 or 70:30).
   - Save the preprocessing steps and transformations as pickle files in a folder named 'project/preprocessing'.

Additional Notes:
   - Ensure the preprocessing steps align with the {user_requirement}.
   - Include any libraries or modules required for implementation.
"""

        # Create the template
        template = PromptTemplate(
            template=prompt, 
            input_variables=['categorical_columns', 'numerical_columns', 'missing_value_columns', 'user_requirement','dataset_path']
        )
        
        # Format the prompt
        formatted = template.format(
            categorical_columns=self.categorical_columns, 
            numerical_columns=self.numerical_columns, 
            missing_value_columns=self.missing_value_columns, 
            user_requirement=self.problem_statement,
            dataset_path=self.dataset_path
        )
        
        # Call the LLM to get the code based on the formatted prompt
        
        counter=0
        while counter  <2:
            output=llm.invoke(formatted).content
            counter+=1
        result=output
        self.preprocessinf_code,self.preprocess_path=gen_to_py(result=result)
       
        logging.info('Preprocessing is Completed')
        
    def Model_building(self):
        logging.info('Starting Model Building Process')
        prompt = '''
        Given the preprocessed code {preprocessing_code}, load the dataset and train  a ML model.Use multiple model and find the model that
        performs the best on the test set. Use the best model to and apply hyperparameter tuning to improve the accuracy of the model. Based on 
        the problem statement {problem_statement}.Save The best Model in the 'project/artifact/Model' folder
        '''
        template = PromptTemplate(
            template=prompt, 
            input_variables=['preprocessing_code','problem_statement']
        )
        formatted = template.format(
            preprocessing_code=self.preprocessinf_code,
            problem_statement=self.problem_statement
        )
        counter=0
        while counter  <2:
            output=llm.invoke(formatted).content
            counter+=1
        self.model_building_code=gen_to_py(output)
        logging.info('Model building  is Completed')
        
    def check_code(self, max_attempts=3):
        logging.info('Checking the generated Codes')
        requirement()
        project_setup()
        # Ensure a log file for recording issues
        
        
        # Get the list of Python files in the 'project' folder
        files = os.listdir('project')
        python_files = [file for file in files if file.endswith('.py')]
        python_files = sorted(python_files, reverse=True)
        
        for file in python_files:
            attempt = 0
            execution = False
            
            while not execution and attempt < max_attempts:
                result = run_generated_code(file)
                
                if result == 'executed':
                    logging.info(f"{file} executed successfully.")
                    execution = True
                else:
                    logging.error(f"Attempt {attempt + 1} failed for {file} with error: {result}")
                    attempt += 1
                    if attempt < max_attempts:
                        logging.info(f"Attempting to correct {file}, attempt {attempt + 1}.")
                        correct_code(file_name=file, error=result)
                        time.sleep(3)
                    else:
                        logging.error(f"Max attempts reached for {file}. Could not resolve the issue.")
                        print(f"Max attempts reached for {file}. Please check the logs for details.")

        print("Code checking process completed. Check logs for details on any issues encountered.")

        
agent=MLAagent(problem_statement='Create a ML model that can help in finding the customer that will default from loan or not',dataset_path=r"C:\Users\bisht\Downloads\archive (83)\Loan_Default.csv",llm=llm)
