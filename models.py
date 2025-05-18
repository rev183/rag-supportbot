from typing import Any
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

# Define a response body model
class ChatResponse(BaseModel):
    answer: str
    source_documents: list[dict[str, Any]]