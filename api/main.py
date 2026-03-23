from fastapi import FastAPI
from pydantic import BaseModel
from agents.graph import app as graph_app, AgentState

api = FastAPI(title="FaultSense API")


# Request schema
class SensorInput(BaseModel):
    sensor_data: dict[str, float]


# Response schema
class DiagnosisOutput(BaseModel):
    status: str
    mean: float
    std: float
    documents: list[str]
    report: str


@api.post("/analyze", response_model=DiagnosisOutput)
def analyze(input: SensorInput):
    initial_state: AgentState = {
        "sensor_data": input.sensor_data,
        "signal_result": {},
        "documents": [],
        "report": "",
        "query": ""
    }

    final_state = graph_app.invoke(initial_state)

    return DiagnosisOutput(
        status=final_state["signal_result"]["status"],
        mean=final_state["signal_result"]["mean"],
        std=final_state["signal_result"]["std"],
        documents=final_state["documents"],
        report=final_state["report"]
    )


@api.get("/health")
def health():
    return {"status": "ok"}