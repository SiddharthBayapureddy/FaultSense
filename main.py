import pandas as pd
from agents.graph import app, AgentState

from dotenv import load_dotenv
load_dotenv()

def load_cmapss(path: str = "./data/train_FD001.txt") -> pd.DataFrame:
    columns = [
        "unit", "cycle", "os1", "os2", "os3",
        *[f"s{i}" for i in range(1, 20)]
    ]
    return pd.read_csv(path, sep=" ", header=None, names=columns).dropna(axis=1)


def run_cli():
    print("Loading CMAPSS data...")
    df = load_cmapss()
    print(df.head())
    print(df.columns.tolist())
    print(df.shape)

    # Test row
    row = df.iloc[0]
    sensor_cols = [f"s{i}" for i in range(1, 18)]
    sensor_data = row[sensor_cols].to_dict()

    print(f"Running agent on unit 1, cycle {int(row['cycle'])}...")

    initial_state: AgentState = {
        "sensor_data": sensor_data,
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
    print(f"\nReport:\n{final_state['report']}")
    print("──────────────────────────────────────")


if __name__ == "__main__":
    run_cli()