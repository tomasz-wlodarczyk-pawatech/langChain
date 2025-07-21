from pydantic import BaseModel
from typing import List

class EventSearch(BaseModel):
    event_id: str

class NoMatch(BaseModel):
    message: str

from pydantic import RootModel

class EventsSearchResponse(RootModel[List[EventSearch]]):
    pass