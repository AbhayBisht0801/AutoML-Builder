import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from load_dotenv import load_dotenv
load_dotenv()
from utils.common import gen_to_py,run_generated_code,correct_code,create_directories,requirement,project_setup
import subprocess
import os
import logging
import time
import shutil
import os
import io
import logging
from io import StringIO
log_stream = StringIO()

# Configure standard logging

llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro')


class MLAagent:
    def __init__(self,problem_statement,dataset_path,logger,llm=llm):
        self.problem_statement = problem_statement
        self.dataset_source=dataset_path
        self.llm=llm
        self.logger=logger
        
    def setup_directory(self):
        self.logger.info('Setting up directory and redirecting Dataset')
        create_directories(['project'])
        data_dir=os.path.join('project','data')
        create_directories([data_dir])
        if self.dataset_source.endswith('csv')==True:
            shutil.move(self.dataset_source,os.path.join(data_dir,'data.csv'))
        else:
            shutil.move(self.dataset_source,os.path.join(data_dir,'data.xlsx'))
        self.dataset_path=os.path.join(data_dir,os.listdir(data_dir)[0])
        self.logger.info(f'Setting completed .Dataset moved to {self.dataset_path}')
        
    def data_insights(self):
        self.logger.info(' Collecting Data Insights')
        
        if self.dataset_path.endswith('csv')==True:
            df=pd.read_csv(self.dataset_path)
        else:
            df=pd.read_excel(self.dataset_path)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        numeric_columns = list(df.select_dtypes(include=numerics).columns)
        categorical_columns = [i for i in df.columns if i not in numeric_columns]
        categorical_columns_features = {col: df[col].unique().tolist() for col in categorical_columns}

        missing_value=(df.isnull().sum()*100)/df.shape[0]
        self.categorical_columns=categorical_columns
        self.numerical_columns=numeric_columns
        self.categorical_columns_features=categorical_columns_features
        self.missing_value_columns=missing_value.to_dict()
        self.logger.info('Data Insights Collected')

    def preprocessing(self):
        self.logger.info('Starting the  Preprocessing of the data')
        prompt = """
Using the input details provided below, generate Python code for preprocessing the dataset:

Input Information:
- User Requirement: {user_requirement} (Describes the context and objective of the task).
- Dataset Path: {dataset_path} (Path to the dataset file
- Numerical Columns: {numerical_columns} (List of numerical features in the dataset).
- Categorical Columns: {categorical_columns} and their features {categorical_columns_features} (List of categorical features and dictionary with their features in the dataset).
- Columns with Missing Values: {missing_value_columns} (Columns that contain missing values).

Instructions for Preprocessing:

1. Handle Missing Values:
   - If the dataset contains missing values, address them based on the context of the {user_requirement}. Use appropriate imputation methods, or leave them as is if justified.
   - If there are no missing values, skip this step.

2. Remove Irrelevant Columns:
   - Understand the problem_statement and drop columns that do not contribute to achieving the {user_requirement}.
   - For example, irrelevant columns could include unique identifiers like "customerId" or personal details like "name".
   - Irrelevant columns should be identified from {numerical_columns} and {categorical_columns}.

3. Data Cleaning:
   - Perform additional cleaning steps such as removing duplicate records, handling outliers, or correcting inconsistent data formats.

4. Encode Categorical Columns:
   - Convert {categorical_columns} into numeric representations using appropriate encoding techniques based on the {categorical_columns_features} to decide if to use One-Hot Encoding or Label Encoding.
   - Do not include any columns identified as irrelevant in this step.

5. Scale Numeric Columns:
   - Normalize {numerical_columns} using a suitable scaling method such as StandardScaler or MinMaxScaler to optimize model performance.

6. Split Dataset:
   - Split the dataset into training and testing subsets based on an appropriate ratio (e.g., 80:20 or 70:30).
   - Save the preprocessing steps and transformations as pickle files in a folder named 'project/preprocessing'.

Additional Notes:
    - Include all libraries or modules required for implementation.
   - Ensure the preprocessing steps align with the {user_requirement} and code generated is not encapulated inside a function.
   -Save the train test split data in preprocessing folder so that it can be later used for training.
   - Create the specific directory before saving the files.
   

"""
        try:
            # Create the template
            template = PromptTemplate(
                template=prompt, 
                input_variables=['categorical_columns', 'numerical_columns', 'missing_value_columns', 'user_requirement','dataset_path','categorical_columns_features']
            )
            
            # Format the prompt
            formatted = template.format(
                categorical_columns=self.categorical_columns, 
                numerical_columns=self.numerical_columns, 
                missing_value_columns=self.missing_value_columns, 
                user_requirement=self.problem_statement,
                dataset_path=self.dataset_path,
                categorical_columns_features=self.categorical_columns_features
            )
            
            # Call the LLM to get the code based on the formatted prompt
            print('Hello')
            counter=0
            while counter  <2:
                output=llm.invoke(formatted).content
                counter+=1
            result=output
            self.preprocessinf_code,self.preprocess_path=gen_to_py(result=result,file_name='preprocessing.py')
        
            self.logger.info('Preprocessing is Completed')
        except Exception as e:
            self.logger.error(f"Error in Preprocessing.retrying again.")
            self.preprocessinf_code,self.preprocess_path=gen_to_py(result=result)
            self.logger.info('Preprocessing is Completed')



    def Model_building(self):
        self.logger.info('Starting Model Building Process')
        prompt = '''
Using the provided {preprocessing_code}, generate a comprehensive Python pipeline for training, evaluating, and optimizing machine learning models. The pipeline should perform the following steps:

Dataset Loading:
Extract file paths for pre-split datasets (X_train, y_train, X_test, y_test) from the given preprocessing_code. Load these datasets into memory for training and evaluation.

Model Training and Evaluation:
Train multiple machine learning models (e.g., Random Forest, Logistic Regression, XGBoost) and evaluate their performance using appropriate metrics (e.g., accuracy, F1-score, or another relevant metric). Identify the best-performing model based on test set results.

Hyperparameter Tuning:
Enhance the accuracy of the best-performing model through hyperparameter tuning (e.g., grid search or randomized search). Reevaluate the optimized model to confirm improvements.



Model Saving:
Save the best-performing, optimized model to the directory project/artifact/Model. Ensure this directory is created dynamically if it does not already exist.

Production-Ready Standards:
Use try-except blocks where applicable to handle potential exceptions gracefully.

Additional Requirements:

Use Python libraries such as scikit-learn for model training and evaluation.
Optimize for scalability and maintainability in a production setting.
Provide example output logs in your implementation.
Clearly align the methodology with the given {problem_statement} to ensure relevance to the problem context.
Deliverable:
Generate Python code for the model training as described above.


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
        self.model_building_code,self.preprocess_path1=gen_to_py(output,file_name='model_training.py')
        self.logger.info('Model building  is Completed')
        
    def check_code(self, max_attempts=3):
        self.logger.info('Setting Up the environment and installing the required library')
        requirement()
        project_setup()
        # Ensure a log file for recording issues
        self.logger.info('Setup Completed')

        
        # Get the list of Python files in the 'project' folder
        self.logger.info('Checking the generated Codes')
        python_files=['preprocessing.py','model_training.py']
        
        for file in python_files:
            self.logger.info(f'Checking {file} code ')
            attempt = 0
            execution = False
            
            while not execution and attempt < max_attempts:
                result = run_generated_code(file)
                
                if result == 'executed':
                    self.logger.info(f"{file} executed successfully.")
                    execution = True
                else:
                    self.logger.info.error(f"Attempt {attempt + 1} failed for {file} with error: {result}")
                    attempt += 1
                    if attempt < max_attempts:
                        self.logger.info(f"Attempting to correct {file}, attempt {attempt + 1}.")
                        correct_code(file_name=file, error=result)
                        time.sleep(3)
                    else:
                        self.logger.error(f"Max attempts reached for {file}. Could not resolve the issue.")
                        print(f"Max attempts reached for {file}. Please check the logs for details.")

        self.logger.info('The Process is Completed.Your Project Folder is available in Project Creation')
    def execute_all(self):
        self.setup_directory()
        self.data_insights()
        self.preprocessing()
        self.Model_building()
        self.check_code()
        self.logger.info('The Process is Completed.Your Project Folder is available in Project Creation')
        
# agent=MLAagent(problem_statement='Create a ML model that helps to find if a person has diabetes or not ',dataset_path=r"C:\\Users\\bisht\\Downloads\\archive (84)\\diabetes.csv",logger=logger)

    