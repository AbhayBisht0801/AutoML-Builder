import os
import pandas as pd

from pydantic import BaseModel,Field
import pandas as pd
from pandasai import Agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
import ast
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
import time
from langchain.agents import AgentExecutor, create_tool_calling_agent
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from load_dotenv import load_dotenv
load_dotenv()

llm=ChatGoogleGenerativeAI(model='gemini-1.0-pro')
# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)

    
class CodeOutput(BaseModel):
    code: str =Field(description='Code of the preprocessing')
    description: str =Field(description='Explanation of Preprocessing Code')
    filename: str=Field(description='Give the respective filename')
    
def parse_output(response):
    parser=PydanticOutputParser(pydantic_object=CodeOutput)
    print(parser.get_format_instructions())
    code_parser_template= """Parse the response from a previous LLM into a 'description' of what was done in the process and a string of valid 'code', 
                                also come up with a 'filename' as the process name this could be saved as that doesnt contain special characters. 
                                Here is the response: {response}. You should parse this in the following JSON Format: """
    json_prompt_tmpl=PromptTemplate(template=code_parser_template,input_variables=['response'],output_parser=parser)
    output=llm.invoke(json_prompt_tmpl.format(response=response))
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
def gen_to_py(result):
    parsed=parse_output(result)
    code_2_func=code_to_function(code=parsed['code'])
    code_to_file(code=code_2_func, file_name=parsed['filename']+'.py')
    return parsed['code']








