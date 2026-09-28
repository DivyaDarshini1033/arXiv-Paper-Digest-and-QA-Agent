import operator
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import json
import arxiv

from tools import search_arxiv, download_pdf, parse_pdf, setup_vector_db

class AgentState(TypedDict):
    query: str
    query_type: str # "id" or "topic"
    candidate_papers: List[Dict]
    selected_paper: Optional[Dict]
    pdf_path: str
    parsed_text: str
    vector_db_dir: str
    collection_name: str
    briefing: str

# Define pydantic models for structured output
class QueryTypeParser(BaseModel):
    query_type: str = Field(description="Must be either 'id' if the query looks like an arXiv ID (e.g. 2305.10601), or 'topic' if it's a general topic.")

class RankedPaper(BaseModel):
    selected_arxiv_id: str = Field(description="The arxiv_id of the most relevant paper.")
    reason: str = Field(description="Brief reason for selection.")

def parse_query_node(state: AgentState):
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
    # Use structured output
    llm_with_structured_output = llm.with_structured_output(QueryTypeParser)
    
    result = llm_with_structured_output.invoke([
        SystemMessage(content="Determine if the user's query is an explicit arXiv ID or a general topic search. arXiv IDs look like 2401.00001 or 1706.03762v5."),
        HumanMessage(content=state["query"])
    ])
    
    return {"query_type": result.query_type}

def fetch_arxiv_node(state: AgentState):
    query = state["query"]
    query_type = state["query_type"]
    
    if query_type == "id":
        # If it's an ID, we just search for that specific ID
        # arxiv library uses id_list for specific IDs
        client = arxiv.Client()
        search = arxiv.Search(id_list=[query.split('v')[0] if 'v' in query else query])
        results = []
        for result in client.results(search):
            arxiv_id = result.entry_id.split('/')[-1]
            results.append({
                "entry_id": result.entry_id,
                "arxiv_id": arxiv_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url,
                "primary_category": result.primary_category
            })
        return {"candidate_papers": results, "selected_paper": results[0] if results else None}
    else:
        results = search_arxiv(query, max_results=5)
        return {"candidate_papers": results}

def rank_papers_node(state: AgentState):
    # Only run if we have multiple candidate papers and no selected paper yet
    if state.get("selected_paper"):
        return {}
        
    candidates = state["candidate_papers"]
    if not candidates:
        raise ValueError("No papers found for the given query.")
        
    if len(candidates) == 1:
        return {"selected_paper": candidates[0]}
        
    # Format candidates for LLM
    candidates_str = ""
    for p in candidates:
        candidates_str += f"ID: {p['arxiv_id']}\nTitle: {p['title']}\nAbstract: {p['summary']}\n\n"
        
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
    llm_with_structured_output = llm.with_structured_output(RankedPaper)
    
    prompt = f"Given the user query: '{state['query']}', select the most relevant paper from the following list:\n\n{candidates_str}"
    
    result = llm_with_structured_output.invoke([
        SystemMessage(content="You are an expert researcher. Rank the papers and return the ID of the most relevant one."),
        HumanMessage(content=prompt)
    ])
    
    selected = next((p for p in candidates if p['arxiv_id'] == result.selected_arxiv_id), candidates[0])
    return {"selected_paper": selected}

def download_and_parse_node(state: AgentState):
    paper = state.get("selected_paper")
    if not paper:
        raise ValueError("No paper selected to download.")
        
    pdf_path = download_pdf(paper["pdf_url"], paper["arxiv_id"])
    if not pdf_path:
        raise Exception("Failed to download PDF.")
        
    text = parse_pdf(pdf_path)
    if not text:
        raise Exception("Failed to parse PDF. It might be malformed or scanned.")
        
    return {"pdf_path": pdf_path, "parsed_text": text}

def chunk_and_embed_node(state: AgentState):
    text = state["parsed_text"]
    paper = state["selected_paper"]
    
    persist_dir, collection_name = setup_vector_db(text, paper["arxiv_id"])
    return {"vector_db_dir": persist_dir, "collection_name": collection_name}

def generate_briefing_node(state: AgentState):
    paper = state["selected_paper"]
    text = state["parsed_text"]
    
    # We take the first 10000 characters to generate the briefing to fit context window
    # and to focus on Abstract/Introduction/Conclusion
    intro_text = text[:15000]
    
    prompt = f"""
    Generate a structured executive briefing for the following paper.
    
    Title: {paper['title']}
    Authors: {', '.join(paper['authors'])}
    Date: {paper['published']}
    ID: {paper['arxiv_id']}
    Link: {paper['pdf_url']}
    
    Paper Content (truncated):
    {intro_text}
    
    Your output MUST be in Markdown and contain EXACTLY the following sections:
    - **Title, Authors, arXiv ID, Publish Date, Link** (Format nicely at the top)
    - **Summary** (1-paragraph plain-English summary: "why this paper matters")
    - **Problem Statement**
    - **Method/Approach** (bullet points)
    - **Key Results/Claims** (bullet points)
    - **Explicit Limitations** (Do not skip this. If none are explicitly stated in the provided text, infer likely limitations and state they are inferred)
    - **Suggested Follow-up Questions** (3-4 questions the user could ask the QA agent)
    """
    
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.2)
    response = llm.invoke([
        SystemMessage(content="You are an expert AI research analyst. Provide a concise, highly accurate executive briefing."),
        HumanMessage(content=prompt)
    ])
    content = response.content
    if isinstance(content, list):
        briefing = "\n".join(str(c.get('text', c)) if isinstance(c, dict) else str(c) for c in content)
    else:
        briefing = str(content)
    
    return {"briefing": briefing}

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("fetch_arxiv", fetch_arxiv_node)
    workflow.add_node("rank_papers", rank_papers_node)
    workflow.add_node("download_and_parse", download_and_parse_node)
    workflow.add_node("chunk_and_embed", chunk_and_embed_node)
    workflow.add_node("generate_briefing", generate_briefing_node)
    
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "fetch_arxiv")
    workflow.add_edge("fetch_arxiv", "rank_papers")
    workflow.add_edge("rank_papers", "download_and_parse")
    workflow.add_edge("download_and_parse", "chunk_and_embed")
    workflow.add_edge("chunk_and_embed", "generate_briefing")
    workflow.add_edge("generate_briefing", END)
    
    return workflow.compile()
