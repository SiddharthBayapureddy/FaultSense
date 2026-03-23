import numpy as np
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI

import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Setting up embedding function
embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./knowledge/chroma_store")
collection = chroma_client.get_or_create_collection(
    name = "maintaince_docs",
    embedding_function = embedder
)


# Pydantic schemas for tool inputs
class SignalProcessorInput(BaseModel):
    sensor_data: dict[str, float] = Field(description="Sensor readings from a single CMAPSS row")

class RAGRetrieverInput(BaseModel):
    query: str = Field(description="Natural language query about the detected fault")


# Defining tools


# Signal Processor
@tool(args_schema=SignalProcessorInput)
def signal_processor(sensor_data: dict[str, float]) -> dict:
    """Analyzes sensor readings and detects anomalies"""
    values = np.array(list(sensor_data.values()))
    mean = np.mean(values)
    std = np.std(values)

    if std > 2.0:   # Sensors wildly inconsistent
        status = "critical"
    elif std > 1.0: # Errors
        status = "degraded"
    else: # Alright
        status = "normal"
    
    return {
        "status": status,
        "mean": mean,
        "std": std
    }


# Rag retriever
# Takes a natural langauge query (generated from signal result), search chromadb for relavant information and returns top matches

@tool(args_schema=RAGRetrieverInput)
def rag_retriever(query: str) -> dict:
    """Retrieves relevant maintenance documents based on a query."""
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    return {
        "documents": results["documents"][0],
        "distances": results["distances"][0]
    }



# Report Writer
# Takes signal result + retrieved docs, calls Mistral to generate a structured diagnosis report

class ReportWriterInput(BaseModel):
    status: str = Field(description="Equipment status: normal, degraded or critical")
    mean: float = Field(description="Mean of sensor readings")
    std: float = Field(description="Standard deviation of sensor readings")
    documents: list[str] = Field(description="Relevant maintenance manual excerpts")

load_dotenv()
llm = ChatMistralAI(model="mistral-large-latest")

@tool(args_schema=ReportWriterInput)
def report_writer(status: str, mean: float, std: float, documents: list[str]) -> dict:
    """Generates a structured diagnosis report based on signal analysis and retrieved documents."""

    context = "\n".join(documents)

    prompt = f"""
    You are a predictive maintenance expert.
    
    Sensor Analysis:
    - Status: {status}
    - Mean: {mean}
    - Std Deviation: {std}
    
    Relevant Manual Excerpts:
    {context}
    
    Generate a structured diagnosis report with:
    1. Fault Type
    2. Severity
    3. Recommended Action
    """
    response = llm.invoke(prompt)
    return {"report": response.content}