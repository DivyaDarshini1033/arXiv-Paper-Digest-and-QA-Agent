import os
import argparse
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

from agent import build_graph, AgentState
from tools import get_retriever
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
console = Console()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def qa_loop(collection_name: str, persist_dir: str):
    console.print("\n[bold green]Entering QA Mode.[/bold green] Type 'exit' or 'quit' to leave.")
    
    retriever = get_retriever(collection_name, persist_dir)
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)
    
    template = """Use the following pieces of context from an academic paper to answer the question at the end.
    If the answer is not contained in the provided context, you MUST state exactly: "I cannot answer this based on the provided paper." Do NOT try to answer based on outside knowledge.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    while True:
        try:
            question = console.input("[bold blue]Ask a question > [/bold blue]")
            if question.lower() in ['exit', 'quit']:
                break
                
            if not question.strip():
                continue
                
            console.print("[dim]Retrieving and generating answer...[/dim]")
            answer = rag_chain.invoke(question)
            
            console.print("\n[bold magenta]Answer:[/bold magenta]")
            console.print(Markdown(answer))
            console.print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error during QA:[/bold red] {e}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous arXiv Paper Digest & QA Agent")
    parser.add_argument("query", type=str, help="Research topic or specific arXiv ID (e.g. '2305.10601')")
    args = parser.parse_args()
    
    if not os.getenv("GEMINI_API_KEY"):
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY not found in .env file.")
        return

    console.print(f"[bold cyan]Starting workflow for query:[/bold cyan] {args.query}")
    
    graph = build_graph()
    current_state = {"query": args.query}
    
    try:
        console.print("[dim]Running LangGraph pipeline...[/dim]")
        for s in graph.stream(current_state, stream_mode="updates"):
            for node_name, node_state in s.items():
                console.print(f"[dim]Finished node: {node_name}[/dim]")
                
                # Update current_state with the new values
                if node_state and isinstance(node_state, dict):
                    current_state.update(node_state)
                
                if node_state and node_name == "parse_query":
                    console.print(f" -> Query interpreted as: [bold]{node_state['query_type']}[/bold]")
                elif node_state and node_name == "fetch_arxiv":
                    candidates = node_state.get('candidate_papers', [])
                    console.print(f" -> Found {len(candidates)} candidate paper(s)")
                    if node_state.get('selected_paper'):
                        console.print(f" -> Selected paper: [bold]{node_state['selected_paper']['title']}[/bold]")
                elif node_state and node_name == "rank_papers" and "selected_paper" in node_state:
                    console.print(f" -> Ranked and selected paper: [bold]{node_state['selected_paper']['title']}[/bold]")
                elif node_state and node_name == "download_and_parse":
                    console.print(f" -> Downloaded and parsed PDF to {node_state['pdf_path']}")
                elif node_state and node_name == "chunk_and_embed":
                    console.print(f" -> Embedded chunks into local Chroma DB ({node_state['collection_name']})")
                elif node_state and node_name == "generate_briefing":
                    console.print("\n[bold yellow]Executive Briefing:[/bold yellow]")
                    console.print(Markdown(node_state["briefing"]))
                    
    except Exception as e:
        console.print(f"[bold red]Pipeline Error:[/bold red] {e}")
        return
        
    # Enter QA loop if we successfully created the vector db
    if "collection_name" in current_state and "vector_db_dir" in current_state:
        qa_loop(current_state["collection_name"], current_state["vector_db_dir"])
    else:
        console.print("[bold red]Failed to reach QA stage.[/bold red]")

if __name__ == "__main__":
    main()
