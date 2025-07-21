from typing import List, Union

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

from rag.Event import Event, NoMatch, EventsResponse
from rag.Event_Search import EventsSearchResponse

load_dotenv()
parser = PydanticOutputParser(pydantic_object=EventsSearchResponse)

prompt = PromptTemplate.from_file(
    "rag/prompt_search.txt",
    input_variables=["user_query", "events"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

llm_chain_search: RunnableSequence = prompt | llm | parser
