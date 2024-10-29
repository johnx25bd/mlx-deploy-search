from pydantic import BaseModel
from typing import List, Union

class QueryRequest(BaseModel):
    query: str

class DocumentResponse(BaseModel):
    rel_docs: List[str]
    rel_docs_sim: List[Union[float, int]]
