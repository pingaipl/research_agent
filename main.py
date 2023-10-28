
import os
from dotenv import load_dotenv
import json
import requests
from bs4 import BeautifulSoup

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory

from pydantic import BaseModel, Field
from typing import Type

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from uvicorn import run

load_dotenv()

serper_api_key = os.environ['SERPER_API_KEY']
browserless_api_key = os.environ['BROWSERLESS_API_KEY']


# Tool 1 : Serach Tool
def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    # print(response.text)
    return response.text


# Tool2 : Scrape Tool
def scrape_website(objective:str, url:str):
    # this tool scrapes the website and also summarizes the content if content is more than 1000 characters
    # Objective is the objective given by user for AI agent. Scraping happens till AI belives that the objective is achived
    
    print("Scraping website...")
    
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

        # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)

        if len(text) > 1000:
            output = summarize(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")
    
# Supporting functionality - summerizer 
# this is map-reduce summerizer
def summarize(objective:str, content:str):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n'], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    
    map_prompt = """
        Provide a precise summary of given context for the given objective.
        Objective : {objective}
        Text : {text}
        Summary : 
    """
    map_prompt_template = PromptTemplate(
        input_variables=["text", "objective"],
        template=map_prompt
    )
    
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt = map_prompt_template,
        verbose=False
    )
    
    output = summary_chain.run(input_documents = docs, objective=objective)
    return output

# To use the function scrape_website as tool, we need to define a toolwrapper for this function

class ScrapeWebsiteInput(BaseModel):
    objective:str = Field(description="The objective and task that user give to agent")
    url:str = Field(description="the url of the website to be scraped")

class ScrapeWebSiteTool(BaseTool):
    name= "scrape_website"
    description= "useful when you want to scrape data from website url by passing both url and objective to function"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective:str, url:str):
        return scrape_website(objective, url)

    def _arun(self, url:str):
        raise NotImplementedError("Not implemeted")


# Create langchain agent with tools defined above
tools = [
    Tool(
        name = "search",
        func = search,
        description="ueseful when information like current event, data needed from internet. Provide precise query for search"
    ),
    ScrapeWebSiteTool()
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iterations
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages" : [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message 
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory",
    return_messages=True,
    llm=llm,
    max_token_limit=1000
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory
)

app = FastAPI()
class Query(BaseModel):
    query: str

@app.post('/')
async def research(query: Query):
    content = agent({"input": query})
    print(content)
    return content



# if __name__ == "__main__":
#     run(app)
#     # search(query)
