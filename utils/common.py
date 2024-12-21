import os
import pandas as pd
import sys
from pydantic import BaseModel,Field
import pandas as pd
from langchain_core.output_parsers.pydantic import PydanticOutputParser
import ast
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
import time
from langchain.agents import AgentExecutor, create_tool_calling_agent
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from load_dotenv import load_dotenv
from utils.modules import moduless,python_inbuilt_modules
load_dotenv()
import re
import subprocess
llm=GoogleGenerativeAI(model='gemini-1.0-pro')
import os
import re
print (moduless)

# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
def project_setup():
    # Define the environment and requirements paths
    env_name = os.path.join('project', 'venv')  # Environment inside 'project' folder
    requirements_file = os.path.join('project', 'requirements.txt')  # Assuming requirements.txt is inside 'project'

    # Create a virtual environment inside the 'project' folder
    subprocess.run([sys.executable, "-m", "venv", env_name], check=True)

    # Install dependencies in the created environment
    pip_path = os.path.join(env_name, "Scripts", "pip") if os.name == "nt" else os.path.join(env_name, "bin", "pip")
    subprocess.run([pip_path, "install", "-r", requirements_file], check=True)

    
class CodeOutput(BaseModel):
    code: str =Field(description='Code of the preprocessing')
    description: str =Field(description='Explanation of Preprocessing Code')
    filename: str=Field(description='Give the respective filename')
    
def parse_output(response):
    parser=PydanticOutputParser(pydantic_object=CodeOutput)
    print(parser.get_format_instructions())
    code_parser_template= """Parse the response from a previous LLM into a 'description' of what was done in the process and a string of valid code in  'code' should not contain any explaination, 
                                also come up with a 'filename' from the process_name and save it in project folder this could be saved as that doesnt contain special characters. 
                                Here is the response: {response}. You should parse this in the following JSON Format: """
    json_prompt_tmpl=PromptTemplate(template=code_parser_template,input_variables=['response'],output_parser=parser)
    output=llm.invoke(json_prompt_tmpl.format(response=response))
    result=output
    print(result)
    output=result.replace('```json','').replace('```','')
    return ast.literal_eval(output)
def code_to_function(code):
    template='''Convert the given code {code}  into  function  with saving preprocessing model in pickle with their respective name so
      that it can be used during model prediction.put the explantion string in #  comments'''
    prompt=PromptTemplate(template=template,input_variables=['code'])
    return llm.invoke(prompt.format(code=code))
def code_to_file(code,name):
    if '.py' in name:
        file_name=name
    else:
        file_name=name+'.py'
    with open(file_name, "w") as f:
        f.write(code.replace('```python','').replace('```',''))
        f.close()
    print('completed')

def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
import os

def gen_to_py(result, path='project',file_name=None):
    error = True
    while error:
        try:
            # Parse the result and extract the code
            parsed = parse_output(result)
            
            # Try writing the code to the file
            code_to_file(code=parsed['code'], name=os.path.join(path, file_name))
            
            # If no exception is raised, set error to False and return
            error = False
            return parsed['code'], file_name
        
        except Exception as e:
            # Log or handle the error if needed
            print(f"Error occurred: {e}. Retrying...")
            # Optionally, you can add a delay or retry limit if needed



        





def run_generated_code(filename):
    try:
        env_name = os.path.join('project', 'venv')  # Environment inside 'project' folder
        file = os.path.join('project', filename)  # Assuming the script is inside 'project'

        # Create a virtual environment inside the 'project' folder
        subprocess.run([sys.executable, "-m", "venv", env_name], check=True)

        # Run the Python file using the environment's Python interpreter
        python_path = os.path.join(env_name, 'Scripts', 'python') if sys.platform == 'win32' else os.path.join(env_name, 'bin', 'python')
        result = subprocess.run([python_path, file], check=True, stderr=subprocess.PIPE, text=True)
        return 'executed'
    except subprocess.CalledProcessError as e:
        # Return any error that occurred during execution
        return f"Error: {e.stderr}"

def correct_code(file_name,error):
    file='project'
    with open(os.path.join(file,file_name), "r") as f:
        code=f.read()
        f.close()
    template='''Correct the {code} getting following {error} make changes in code accordingly. Generate the entire corrected  code.'''
    prompt=PromptTemplate(template=template,input_variables=['code','error'])
    result=llm.invoke(prompt.format(code=code,error=error))
    gen_to_py(result=result,file_name=file_name)

def requirement(foldername='project'):
    pattern = r'^import (\w+)|^from (\w+)'

    # List to store the names of imported modules
    library = []

    # Get list of all files in the directory and filter out Python files
    files = os.listdir(foldername)
    filter_file = [i for i in files if i.endswith('.py')]

    # Read each Python file line by line
    for filename in filter_file:
        # Construct full file path
        filepath = os.path.join(foldername, filename)
        with open(filepath, 'r') as f:
            for line in f:
                # Apply the regex to each line
                match = re.match(pattern, line)
                
                if match:
                    # Add non-empty matched groups to the library list
                    library.extend([name for name in match.groups() if name])

    # Remove duplicates and join modules with newline
    lib=[moduless[k] for k in library if k in moduless ]


    libraries = '\n'.join(set(lib))
    

    print(libraries)
    # Write the modules to 'requirements.txt'
    with open(os.path.join(foldername, 'requirements.txt'), 'w') as f:
        f.write(libraries)


    








