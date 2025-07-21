from typing import List, Union

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

from rag.Event import Event, NoMatch, EventsResponse

load_dotenv()
parser = PydanticOutputParser(pydantic_object=EventsResponse)

prompt = PromptTemplate.from_file(
    "rag/prompt.txt",
    input_variables=["user_query", "events"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

llm_chain: RunnableSequence = prompt | llm | parser
