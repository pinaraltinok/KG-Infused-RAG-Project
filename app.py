from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from query_runner import answer_question


app = FastAPI(title="SocialRAG", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = Field("kg_rag")  # kg_rag|vanilla_rag|no_retrieval
    llm: str = Field("ollama")  # ollama|gemini
    domain: str | None = None
    rounds: int = 3
    seed_k: int = 5
    entities_per_round: int = 10
    wiki_lang: str = "tr"
    passage_k: int = 8


@app.post("/api/ask")
def api_ask(req: AskRequest):
    return answer_question(
        req.question,
        mode=req.mode,
        llm=req.llm,
        domain=req.domain,
        rounds=req.rounds,
        seed_k=req.seed_k,
        entities_per_round=req.entities_per_round,
        wiki_lang=req.wiki_lang,
        passage_k=req.passage_k,
    )


@app.post("/api/answer", response_class=PlainTextResponse)
def api_answer(req: AskRequest):
    out = answer_question(
        req.question,
        mode=req.mode,
        llm=req.llm,
        domain=req.domain,
        rounds=req.rounds,
        seed_k=req.seed_k,
        entities_per_round=req.entities_per_round,
        wiki_lang=req.wiki_lang,
        passage_k=req.passage_k,
    )
    # Standardize to plain text single-line answer
    ans = out.get("answer") if isinstance(out, dict) else str(out)
    if not ans:
        ans = ""
    return str(ans).strip().replace("\r", " ").replace("\n", " ").strip()


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    # frontend served as static file
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

