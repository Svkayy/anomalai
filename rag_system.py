"""
RAG System for generating formal safety reports from observations
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import google.generativeai as genai
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings
    from supabase import create_client
    
    # Initialize components
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    # Set up Gemini
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Set up Ollama Embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Set up Supabase client and vector store
    supabase_client = create_client(supabase_url, supabase_key)
    vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name="policy_embeddings_vs",
        query_name="match_documents"
    )
    
    RAG_AVAILABLE = True
    
except ImportError as e:
    print(f"RAG system dependencies not available: {e}")
    RAG_AVAILABLE = False
    gemini_model = None
    vector_store = None
    supabase_client = None


def store_policies() -> int:
    """
    Function to store policies from Supabase to the vector store.
    Returns 1 on success, 0 on failure.
    """
    if not RAG_AVAILABLE:
        print("[rag_system.py][store_policies] RAG system not available")
        return 0
        
    try:
        # Read all policies documents from the "policy_embeddings" table
        response = supabase_client.table("policy_embeddings").select("*").execute()

        if response.data:
            print(f"[rag_system.py][store_policies] First document structure: {response.data[0]}")
        else:
            print("[rag_system.py][store_policies] No documents found in table")
            return 0
        
        # Store all fetched and converted documents into an array
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
        print(f"[rag_system.py][store_policies] Successfully stored {len(documents)} policies into the vector store.")
        return 1
    except Exception as e:
        print(f"[rag_system.py][store_policies] Error storing policies: {e}")
        return 0


def build_context(query: str, k: int = 10) -> str:
    """
    Use Ollama embeddings to find the k-most relevant documents to a query.
    Returns a string containing the context built from these documents.
    """
    if not RAG_AVAILABLE:
        return ""
        
    try:
        relevant_docs = vector_store.similarity_search(query, k=k)
        # Concatenate the content of the relevant documents into a single context string
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return context
    except Exception as e:
        print(f"[rag_system.py] Error building context: {e}")
        # Fallback: return basic OSHA context if vector search fails
        return """
        OSHA General Industry Standards (29 CFR 1910):
        - 1910.132: General Requirements for Personal Protective Equipment
        - 1910.22: Walking-Working Surfaces
        - 1910.23: Guarding Floor and Wall Openings and Holes
        - 1910.24: Fixed Industrial Stairs
        - 1910.25: Portable Wood Ladders
        - 1910.26: Portable Metal Ladders
        - 1910.27: Fixed Ladders
        - 1910.28: Safety Requirements for Scaffolding
        - 1910.29: Manually Propelled Mobile Ladder Stands and Platforms
        - 1910.30: Other Working Surfaces
        """


def generate_formal_safety_report(observations_data: Dict) -> str:
    """
    Generate a formal safety report using RAG system based on observations data.
    
    Args:
        observations_data: Dictionary containing structured observations data
        
    Returns:
        Generated formal safety report text
    """
    if not RAG_AVAILABLE:
        return "RAG system not available - cannot generate formal report"
    
    try:
        # Extract key information from observations for the query
        query_parts = []
        
        # Add video analysis summary
        if 'video_analysis_summary' in observations_data:
            summary = observations_data['video_analysis_summary']
            query_parts.append(f"Video analysis: {summary.get('total_frames_analyzed', 0)} frames analyzed, {summary.get('unsafe_frames_count', 0)} unsafe frames")
        
        # Add frame analyses
        if 'frame_analyses' in observations_data:
            for frame in observations_data['frame_analyses']:
                if 'observations' in frame:
                    for obs in frame['observations']:
                        label = obs.get('label', 'unknown')
                        severity = obs.get('severity', 'unknown')
                        reasons = obs.get('reasons', [])
                        query_parts.append(f"Safety violation: {label} (severity: {severity}) - {', '.join(reasons)}")
        
        # Create comprehensive query
        query = " ".join(query_parts) if query_parts else "workplace safety violations"
        
        # Get relevant context from RAG
        context = build_context(query=query, k=10)
        
        # Create detailed prompt for formal report generation
        prompt = f"""
You are an Occupational Health and Safety expert. Generate a comprehensive formal safety report based on the following workplace safety observations and relevant OSHA regulations.

CONTEXT FROM OSHA REGULATIONS:
{context}

SAFETY OBSERVATIONS DATA:
{json.dumps(observations_data, indent=2)}

INSTRUCTIONS:
1. Generate a professional, formal safety report
2. Reference specific OSHA regulations where applicable
3. Include executive summary, detailed findings, and recommendations
4. Structure the report with clear sections and subsections
5. Be specific about violations and their severity
6. Provide actionable recommendations for each violation
7. Include compliance requirements and timelines
8. Use professional safety report language

FORMAT:
- Executive Summary
- Detailed Findings (with OSHA regulation references)
- Risk Assessment
- Recommendations and Corrective Actions
- Compliance Requirements
- Timeline for Implementation

Please generate a comprehensive formal safety report based on the observations provided.
"""
        
        # Generate the report using Gemini
        response = gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"[rag_system.py] Error generating formal report: {e}")
        return f"Error generating formal safety report: {str(e)}"


def is_rag_available() -> bool:
    """Check if RAG system is available and properly configured"""
    return RAG_AVAILABLE


# Initialize policies when module is imported
if RAG_AVAILABLE:
    try:
        store_policies()
    except Exception as e:
        print(f"[rag_system.py] Error initializing policies: {e}")
