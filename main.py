from typing import List, Literal
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from clinical_trials_retrieval import ClinicalTrialsRetrieval
from cord19_retrieval import Cord19Retrieval

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

clinical_retrieval_system = ClinicalTrialsRetrieval()
cord19_retrieval_system = Cord19Retrieval()

# Input model
class QueryRequest(BaseModel):
    query: str
    query_id: str
    top_n: int = 10
    mode: Literal["tfidf", "word2vec", "hybrid"] = "tfidf"
    dataset: Literal["clinical", "cord19"] = "clinical"
    use_topic_filter: bool = False

# Output models
class Document(BaseModel):
    doc_id: str
    title: str
    summary: str
    description: str
    similarity: float

class QueryResponse(BaseModel):
    query: str
    top_n: int
    mode: str
    retrieved_documents: List[Document]
    matched_count: int
    precision: float
    matched_ids: List[str]

# Unified endpoint
@app.post("/search", response_model=QueryResponse)
def search_query(request: QueryRequest):
    print(request)
    if request.dataset == "clinical":
        retrieval_system = clinical_retrieval_system
    else:
        retrieval_system = cord19_retrieval_system

    if request.mode == "tfidf":
        retrieved_documents = retrieval_system.search(request.query, request.top_n)
    elif request.mode == "word2vec":
        retrieved_documents = retrieval_system.search_word2vec(request.query, request.top_n)
    elif request.mode == "hybrid":
        retrieved_documents = retrieval_system.search_hybrid(
            query=request.query,
            top_n=request.top_n,
            use_topic_filter=request.use_topic_filter
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'tfidf' or 'word2vec'.")

    retrieved_doc_ids = [doc["doc_id"] for doc in retrieved_documents]
    evaluation = retrieval_system.evaluate(request.query_id, retrieved_doc_ids)

    return QueryResponse(
        query=request.query,
        top_n=request.top_n,
        mode=request.mode,
        retrieved_documents=retrieved_documents,
        matched_count=evaluation["matched_count"],
        precision=evaluation["precision"],
        matched_ids=evaluation["matched_ids"]
    )

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})