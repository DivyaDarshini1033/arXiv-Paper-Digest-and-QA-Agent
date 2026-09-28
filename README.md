# Autonomous arXiv Paper Digest & QA Agent

This repository contains a CLI-based AI agent that takes an arXiv ID or a research topic, fetches the most relevant paper, digests its content, generates an executive briefing, and lets you ask questions about it with strict RAG grounding.

## Features
- **Smart Query Parsing**: Distinguishes between specific arXiv IDs (e.g., `2305.10601`) and vague topic searches (e.g., "prompt engineering").
- **Auto-Ranking**: Fetches top 5 papers for a topic search and uses an LLM to select the most relevant one.
- **Robust PDF Parsing**: Downloads and extracts text from PDFs using `PyMuPDF`.
- **Local Vector DB**: Chunks text and stores it in a local `Chroma` database for retrieval.
- **Executive Briefing**: Generates a structured markdown briefing summarizing the problem, method, results, and limitations.
- **Grounded QA Loop**: Strict RAG constraints ensure the agent does not hallucinate answers outside of the paper context.

## Architecture

The agent is built as an explicit state graph using **LangGraph**. The shared state persists across nodes, carrying the paper metadata, parsed text, and vector database references.

### State Graph Nodes:
1. `parse_query`: LLM decides if the query is an ID or a topic.
2. `fetch_arxiv`: Hits the arXiv API to get metadata and PDF links.
3. `rank_papers`: (Topic only) Ranks candidates and selects the best fit.
4. `download_and_parse`: Downloads the PDF to a local `/data` folder and extracts text.
5. `chunk_and_embed`: Chunks the text and embeds it into a local `Chroma` database (in `/chroma_db`).
6. `generate_briefing`: LLM reads the abstract and introduction to generate a markdown briefing.
7. `QA Loop` (Main script): After the graph completes, an interactive loop uses a LangChain retriever and a strict prompt to answer questions.

## Design Decisions & Tradeoffs

1. **LangGraph vs. Linear Chain**: A state graph was chosen over a monolithic prompt chain to handle the conditional logic (e.g., branching for topic vs. ID) cleanly and to modularize the heavy tasks (downloading, parsing, embedding). It makes error handling and state inspection much easier.
2. **PyMuPDF for Parsing**: Chosen for its speed and reliability on academic two-column PDFs compared to basic parsers like `pdfplumber` or `pypdf`. A tradeoff is that highly complex layouts with math formulas might not parse perfectly, but text extraction is sufficient for RAG.
3. **Local Chroma Vector Store**: `Chroma` was selected because it's easy to run locally without external cloud dependencies.
4. **Google Gemini (Free Tier)**: Gemini 1.5 Flash is used for routing and QA due to its speed, while Gemini 1.5 Pro is used for generating the detailed briefing. This balances performance and latency while remaining free. The embedding model (`text-embedding-004`) is also from Google.
5. **Strict Grounding Prompt**: The QA prompt enforces that the model must say "I cannot answer this based on the provided paper" if the context doesn't contain the answer, minimizing hallucination.

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API Key (get one for free at [Google AI Studio](https://aistudio.google.com/))

### 2. Installation
Clone or copy this repository, then create a virtual environment and install dependencies:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configuration
Copy the `.env.example` file to `.env`:
```bash
# Windows:
copy .env.example .env
# Mac/Linux:
# cp .env.example .env
```
Edit `.env` and add your `GEMINI_API_KEY`.

### 4. Running the Agent
Run the main script and provide a query. 

**Example 1: Specific arXiv ID**
```bash
python main.py "2305.10601"
```

**Example 2: Topic Search**
```bash
python main.py "Recent advancements in mechanistic interpretability"
```

Once the executive briefing is printed, you will enter the **QA Mode** where you can ask questions. Type `exit` to quit.
