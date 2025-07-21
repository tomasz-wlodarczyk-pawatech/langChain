from pydantic import BaseModel
from typing import List, Union
from langchain.output_parsers import PydanticOutputParser

class Event(BaseModel):
    event_name: str
    competition: str
    start_time: str
    event_id: str
    markets: List[str]

class NoMatch(BaseModel):
    message: str

from pydantic import RootModel

class EventsResponse(RootModel[List[Event]]):
    pass