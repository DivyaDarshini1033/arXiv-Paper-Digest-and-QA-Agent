import arxiv
import pymupdf  # PyMuPDF
import os
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def search_arxiv(query: str, max_results: int = 5):
    """Search arXiv for papers based on a query."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    try:
        for result in client.results(search):
            # Arxiv IDs are typically the end of the entry_id URL
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
    except Exception as e:
        print(f"Error fetching from arXiv: {e}")
        
    return results

def download_pdf(pdf_url: str, arxiv_id: str, output_dir: str = "data"):
    """Downloads a PDF from a URL."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Extract just the ID if it's a full URL and make it safe for filesystem
    safe_id = arxiv_id.replace("/", "_").replace(":", "_")
    filepath = os.path.join(output_dir, f"{safe_id}.pdf")
    
    if os.path.exists(filepath):
        return filepath
        
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filepath
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None

def parse_pdf(filepath: str):
    """Parses a PDF using PyMuPDF and returns text."""
    try:
        doc = pymupdf.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n\n"
        return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None

def setup_vector_db(text: str, arxiv_id: str, persist_directory: str = "chroma_db"):
    """Chunks text and stores it in ChromaDB."""
    # We use Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": arxiv_id}) for chunk in chunks]
    
    # Store in Chroma
    # We create a safe collection name (alphanumeric and underscores only, must start with letter)
    safe_collection_name = f"arxiv_{arxiv_id.replace('.', '_').replace('-', '_')}"
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=safe_collection_name
    )
    return persist_directory, safe_collection_name

def get_retriever(collection_name: str, persist_directory: str = "chroma_db"):
    """Gets a retriever for a specific Chroma collection."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    return vectorstore.as_retriever(search_kwargs={"k": 10})
