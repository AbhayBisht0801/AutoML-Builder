import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from load_dotenv import load_dotenv
load_dotenv()
from common import gen_to_py

llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro')
class MLAagent:
    def __init__(self,problem_statement,dataset_path):
        self.problem_statement = problem_statement
        self.dataset_path = dataset_path
        self.llm=llm
      
    
    def data_insights(self):
        if self.dataset_path.endswith('csv')==True:
            df=pd.read_csv(self.dataset_path)
        else:
            df=pd.read_excel(self.dataset_path)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        numeric_columns = list(df.select_dtypes(include=numerics).columns)
        categorical_columns = [i for i in df.columns if i not in numeric_columns]
        missing_value=(df.isnull().sum()*100)/df.shape[0]
        self.categorical_columns=categorical_columns
        self.numeric_columns=numeric_columns
        self.missing_value_columns=missing_value.to_dict()
    def preprocessing(self):
        prompt = '''
    Given the {user_requirement}, perform the necessary preprocessing steps based on the csv dataset infomation provided, which includes:

    Numeric Columns: {numerical_columns}
    Categorical Columns: {categorical_feature}
    Columns with Missing Values: {missing_value_columns}
    based on the above info about dataset follow the instructions below for data preprocessing:

    - Handle Missing Values: If the dataset contains missing values , address them according to the context of the problem statement given missing column in {missing_value_columns}. In real-time data, it is not always ideal to impute missing values, so use an appropriate method or leave them if justified. If no missing values exist, proceed to the next steps.

    - Remove Irrelevant Columns: Identify and drop columns  that do not contribute to model performance or if there is irrelevant columns  for  example like customerId,name with respect to customer data.Identify the irrelevant columns from {numerical_columns}
    and {categorical_feature}.

    - Data Cleaning: Perform any additional cleaning steps such as removing duplicates, handling outliers, or correcting data formats as required.

    - Encode Categorical Columns: Use suitable encoding techniques (e.g., One-Hot Encoding or Label Encoding) to convert categorical variables into numeric representations with columns {categorical_feature}.dont select the irrlevant column in encoding which are droped.
    - Scale Numeric Columns: Apply appropriate scaling methods (e.g., StandardScaler, MinMaxScaler) to normalize the {numerical_columns} for optimal model performance.
    - Train_Test_split: Split the Data into Train Test  and save it into a 'project/artifact' folder
        '''
        
        # Create the template
        template = PromptTemplate(
            template=prompt, 
            input_variables=['categorical_feature', 'numerical_columns', 'missing_value_columns', 'user_requirement']
        )
        
        # Format the prompt
        formatted = template.format(
            categorical_feature=self.categorical_columns, 
            numerical_columns=self.numerical_columns, 
            missing_value_columns=self.missing_value_columns, 
            user_requirement=self.problem_statement
        )
        
        # Call the LLM to get the code based on the formatted prompt
        output=llm.invoke(formatted).content
        self.preprocessinf_code=gen_to_py(output)
    def Model_building(self):
        prompt = '''
        Given the preprocessed code {preprocessing_code}, load the dataset and train  a ML model.Use multiple model and find the model that
        performs the best on the test set. Use the best model to and apply hyperparameter tuning to improve the accuracy of the model. Based on 
        the problem statement {problem_statement}.
        '''
        template = PromptTemplate(
            template=prompt, 
            input_variables=['categorical_feature', 'numerical_columns', 'missing_value_columns', 'user_requirement']
        )
        formatted = template.format(
            categorical_feature=self.preprocessinf_code,
            user_requirement=self.problem_statement
        )
        output=llm.invoke(formatted).content
        self.model_building_code=gen_to_py(output)

        

        
agent=MLAagent(problem_statement='Given the dataset create a ML model that find if a person is gonna defualt from loan or not'
               ,dataset_path=r"C:\Users\bisht\Downloads\archive (83)\Loan_Default.csv")
agent.data_insights()
agent.preprocessing()