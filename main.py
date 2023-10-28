
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
from fastapi import Response

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
    print("Summarizing...")
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n'], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    
    map_prompt = """
        Extract the key information for the following text for {objective}. The text is Scraped data from a website so 
        will have a lot of usless information that doesnt relate to this topic, links, other news stories etc.. 
        Only summarise the relevant Info and try to keep as much factual information Intact
        Do not describe what the webpage is, you are here to get acurate and specific information
        Example of what NOT to do: "Investor's Business Daily: Investor's Business Daily provides news and trends on AI stocks and artificial intelligence. They cover the latest updates on AI stocks and the trends in artificial intelligence. You can stay updated on AI stocks and trends at [AI News: Artificial Intelligence Trends And Top AI Stocks To Watch "
        Here is the text:

        "{text}"
        SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        input_variables=["text", "objective"],
        template=map_prompt
    )
    
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt = map_prompt_template,
        verbose=True
    )
    
    output = summary_chain.run(input_documents = docs, objective=objective)
    return output

# To use the function scrape_website as tool, we need to define a toolwrapper for this function

class ScrapeWebsiteInput(BaseModel):
    objective:str = Field(description="The objective and task that user give to agent")
    url:str = Field(description="the url of the website to be scraped")

class ScrapeWebSiteTool(BaseTool):
    name= "scrape_website"
    description= "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
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
    content="""
        You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
        you do not make things up, you will try as hard as possible to gather facts & data to back up the research
        
        Please make sure you complete the objective above with the following rules:
        1/ You should do enough research to gather as much information as possible about the objective
        2/ If there are url of relevant links & articles, you will scrape it to gather more information
        3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
        4/ You should not make things up, you should only write facts & data that you have gathered
        5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
        6/ Always look at the web first
        7/ Output as much information as possible, make sure your answer is at least 500 WORDS
        8/ Be specific about your reasearch, do not just point to a website and say things can be found here, that what you are for
        

        Example of what NOT to do return these are just a summary of whats on the website an nothing specific, these tell the user nothing!!

        1/WIRED - WIRED provides the latest news, articles, photos, slideshows, and videos related to artificial intelligence. Source: WIRED

        2/Artificial Intelligence News - This website offers the latest AI news and trends, along with industry research and reports on AI technology. Source: Artificial Intelligence News
        """
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

from langchain.retrievers.multi_query import MultiQueryRetriever

template = """
    You are a very experienced ghostwriter who excels at writing Twitter threads.
    You will be given a bunch of info below and a topic headline, your job is to use this info and your own knowledge
    to write an engaging Twitter thread.
    The first tweet in the thread should have a hook and engage with the user to read on.

    Here is your style guide for how to write the thread:
    1. Voice and Tone:
    Informative and Clear: Prioritize clarity and precision in presenting data. Phrases like "Research indicates," "Studies have shown," and "Experts suggest" impart a tone of credibility.
    Casual and Engaging: Maintain a conversational tone using contractions and approachable language. Pose occasional questions to the reader to ensure engagement.
    2. Mood:
    Educational: Create an atmosphere where the reader feels they're gaining valuable insights or learning something new.
    Inviting: Use language that encourages readers to dive deeper, explore more, or engage in a dialogue.
    3. Sentence Structure:
    Varied Sentence Lengths: Use a mix of succinct points for emphasis and longer explanatory sentences for detail.
    Descriptive Sentences: Instead of directive sentences, use descriptive ones to provide information. E.g., "Choosing a topic can lead to..."
    4. Transition Style:
    Sequential and Logical: Guide the reader through information or steps in a clear, logical sequence.
    Visual Emojis: Emojis can still be used as visual cues
    5. Rhythm and Pacing:
    Steady Flow: Ensure a smooth flow of information, transitioning seamlessly from one point to the next.
    Data and Sources: Introduce occasional statistics, study findings, or expert opinions to bolster claims, and offer links or references for deeper dives.
    6. Signature Styles:
    Intriguing Introductions: Start tweets or threads with a captivating fact, question, or statement to grab attention.
    Question and Clarification Format: Begin with a general question or statement and follow up with clarifying information. E.g., "Why is sleep crucial? A study from XYZ University points out..."

    Engaging Summaries: Conclude with a concise recap or an invitation for further discussion to keep the conversation going.
    Distinctive Indicators for an Informational Twitter Style:

    Leading with Facts and Data: Ground the content in researched information, making it credible and valuable.
    Engaging Elements: The consistent use of questions and clear, descriptive sentences ensures engagement without leaning heavily on personal anecdotes.
    Visual Emojis as Indicators: Emojis are not just for casual conversations; they can be effectively used to mark transitions or emphasize points even in an informational context.
    Open-ended Conclusions: Ending with questions or prompts for discussion can engage readers and foster a sense of community around the content.

    Last instructions:
    The twitter thread should be between the length of 3 and 10 tweets 
    Each tweet should start with (tweetnumber/total length)
    Dont overuse hashtags, only one or two for entire thread.
    The first tweet, do not place a number at the start.
    When numbering the tweetes Only the tweetnumber out of the total tweets. i.e. (1/9) not (tweet 1/9)
    Use links sparingly and only when really needed, but when you do make sure you actually include them AND ONLY PUT THE LINk, dont put brackets around them. 
    Only return the thread, no other text, and make each tweet its own paragraph.
    Make sure each tweet is lower that 220 chars
    Topic Headline:{topic}
    Info: {info}
    """

prompt = PromptTemplate(
    input_variables=["info","topic"], template=template
)

from langchain.chains import LLMChain

llm = ChatOpenAI()
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    
)

app = FastAPI()
class Query(BaseModel):
    query: str

# class LLMResponse(BaseModel):
#     output: str

@app.post('/')
async def research(query: Query):
    results = agent({"input": query.query})
    actual_content = results['output']
    thread = llm_chain.predict(info = actual_content, topic = query)
    
    return thread

@app.post('/telegram')
async def splitText(text: Query):
    messages = text.query.split("\n\n")
    print(messages)
    messages = [s.replace('. ', '.\n\n') for s in messages]
    print(messages)
    messages = [s.replace('**', '') for s in messages]
    print(messages)

# if __name__ == "__main__":
#     query = input("Enter the query : ")
#     content = agent(query)
#     print(type(content))
#     content 
#     # search(query)
