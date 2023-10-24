import os
from pyexpat import model
from statistics import mode
from dotenv import load_dotenv
import json
import requests
from bs4 import BeautifulSoup

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

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
    print(response.text)
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
        You are an expeienced research agent who can perform research of given context and produce a detailed report. 
        The research has an objective mentioned below. 
        The text given below is your context.  
        Your report should provide introduction, key details, major achievements, significant events etc. 
        The report can include contact details such as office location, address, email IDs or telephone numbers.
        Use list of items aproach when listing items. 
        Objective : {objective}
        Text : {text}
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

if __name__ == "__main__":
    # query = input('Enter a search query: ')
    # search(query)
    test_url = 'https://www.youtube.com/watch?v=5HBGEPbIQGo'
    resp = scrape_website('short essay', test_url)
    print(resp)