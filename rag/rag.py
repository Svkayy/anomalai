# System packages
from dotenv import load_dotenv
import json
import os
# Networking packages
from fastapi import FastAPI, Request
# AI packages
import google.genai as genai
# Vector store and embedding packages
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
# Database packages
from supabase import create_client


# Load environment variables
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Set up Gemini
client = genai.Client(api_key=gemini_key)

# Set up Ollama Embeddings (use medium-sized embeddings model)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Set up DB connection to Supabase and set up vector store instance
supabase = create_client(supabase_url, supabase_key)
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="policy_embeddings_vs",
    query_name="match_documents"
)


"""
    Function to store policies from Supabase to the vector store. Returns 1 on
    success, 0 on failure.
"""
def store_policies() -> int:
    try:
        # Read all policies documents from the "policy_embeddings" table
        response = supabase.table("policy_embeddings").select("*").execute()

        if response.data:
            print(f"[rag.py][store_policies] First document structure: \
                  {response.data[0]}")
        else:
            print("[rag.py][store_policies] No documents found in table")
            return 0
        
        # Store all fetched and converted documented into an array
        documents = []
        for document in response.data:
            doc = Document(
                page_content=document['content'],
                metadata={
                    'id': document['id'],
                    'title': document['title'],
                    'category': document['category']
                }
            )
            documents.append(doc)

        # Add documents to the vector store
        vector_store.add_documents(documents)
        print(f"[rag.py][store_policies] Successfully stored {len(documents)} \
               policies into the vector store.")
        return 1
    except Exception as e:
        print(f"[rag.py][store_policies] Error storing policies: {e}")
        return 0

store_policies()


"""
    Use Ollama embeddings to find the k-most relevant documents to a query.
    Returns a string containing the context built from these documents, fed
    to Gemini for question answering. Default: k = 3.
"""
def build_context(query: str, k: int = 3) -> str:
    try:
        relevant_docs = vector_store.similarity_search(query, k=k)
        # Concatenate the content of the relevant documents into a single context string
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return context
    except Exception as e:
        print(f"[rag.py] Error building context: {e}")
        return ""
    

@app.post(
    "/rag/ask",
    tags=["ask"],
    summary="Ask a question to the RAG system",
    description="Endpoint to ask a question to the RAG system and get a detailed answer."
)
def prompt(query: str) -> str:
    # Get 10 most relevant documents from the vector store
    context = build_context(query=query, k=10)

    prompt = \
        f"""
            You are a Occupational Health and Safety expert. Use the following context to
            answer the question at the end. If you don't know the answer, just say that you
            don't know, don't try to make up an answer. Be concise and to the point. Please
            cite any OHSA regulations you refer to in your answer.

            Here are some relevant contexts to help you answer the question:
            {context}
        """
    
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{prompt} \n\n Question: {query}", 
    )

    print(resp.text)
    return resp.text

prompt("Electrical equipment near a water source supply.")