from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .tools import signal_processor, rag_retriever, report_writer

# The brain

# Graph Struture:

        # START
        #   ↓
        # signal_processor_node   ← processes sensor data
        #   ↓
        # rag_retriever_node      ← queries ChromaDB based on signal result
        #   ↓
        # report_writer_node      ← generates final diagnosis via Mistral
        #   ↓
        # END



# State of the graph
class AgentState(TypedDict):

    sensor_data: dict[str, float]  # CMAPSS row
    signal_result: dict # output of signal_processor
    documents: list[str] # output of rag_retriever
    report: str # output of report_writer
    query: str


# Node Functions build from AgentState -> AgentState

def signal_processor_node(state: AgentState) -> AgentState:
    result = signal_processor.invoke({"sensor_data" : state["sensor_data"]})
    query = f"Maintaince Procedure for {result['status']} equipement with std {result['std']}"

    return {
        "signal_result": result,
        "query":query
    }


def rag_retriever_node(state: AgentState) -> AgentState:
    result = rag_retriever.invoke({"query": state["query"]})
    return {
        "documents": result["documents"]
    }

def report_writer_node(state: AgentState) -> AgentState:
    result = report_writer.invoke({
        "status": state["signal_result"]["status"],
        "mean": state["signal_result"]["mean"],
        "std": state["signal_result"]["std"],
        "documents": state["documents"]
    })
    return {"report": result["report"]}


# Building the graph

graph = StateGraph(AgentState)

# Nodes
graph.add_node("signal_processor", signal_processor_node)
graph.add_node("rag_retriever", rag_retriever_node)
graph.add_node("report_writer", report_writer_node)

# Edges
graph.add_edge(START, "signal_processor")
graph.add_edge("signal_processor", "rag_retriever")
graph.add_edge("rag_retriever", "report_writer")
graph.add_edge("report_writer", END)


app = graph.compile()