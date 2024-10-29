from fastapi import FastAPI
from models.schemas import QueryRequest, DocumentResponse

from services.search import get_docs

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the search API. Use POST /search to send queries."}

@app.post("/search", response_model=DocumentResponse)
async def search(query_request: QueryRequest):
    query = query_request.query
    # query_embedding = preprocess(query)
    rel_docs, urls, distances  = get_docs(query)
    return {
        "rel_docs": rel_docs,
        "urls": urls,
        "rel_docs_sim": distances[0]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
