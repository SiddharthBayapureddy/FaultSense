import pandas as pd
from agents.graph import app, AgentState

from dotenv import load_dotenv
load_dotenv()

def load_cmapss(path: str = "./data/train_FD001.txt") -> pd.DataFrame:
    columns = [
        "unit", "cycle", "os1", "os2", "os3",
        *[f"s{i}" for i in range(1, 18)]
    ]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="python", index_col=False)
    df = df.iloc[:, :22]
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def run_cli():
    print("Loading CMAPSS data...")
    df = load_cmapss()

    # Get last 30 cycles of engine unit 1
    unit_df = df[df["unit"] == 1].sort_values("cycle").tail(30)
    sensor_cols = [f"s{i}" for i in range(1, 18)]
    
    # Single row for signal_processor (latest cycle)
    row = unit_df.iloc[-1]
    sensor_data = row[sensor_cols].to_dict()

    # Window for rul_predictor (last 30 cycles)
    window = unit_df[sensor_cols].values.tolist()

    print(f"Running agent on unit 1, last 30 cycles...")

    initial_state: AgentState = {
        "sensor_data": sensor_data,
        "window": window,
        "signal_result": {},
        "documents": [],
        "report": "",
        "query": ""
    }

    final_state = app.invoke(initial_state)

    print("\n── DIAGNOSIS REPORT ──────────────────")
    print(f"Status   : {final_state['signal_result']['status']}")
    print(f"Mean     : {final_state['signal_result']['mean']:.3f}")
    print(f"Std      : {final_state['signal_result']['std']:.3f}")
    print(f"RUL      : {final_state['rul']} cycles")
    print(f"Urgency  : {final_state['urgency']}")
    print(f"\nReport:\n{final_state['report']}")
    print("──────────────────────────────────────")


if __name__ == "__main__":
    run_cli()