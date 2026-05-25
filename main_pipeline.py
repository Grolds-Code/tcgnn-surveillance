import pandas as pd
import torch
import os
import numpy as np
from backend.graph_engine import TopologyAwareGraphBuilder
from backend.tcgnn_model import TopologyCausalGNN


def generate_synthetic_data(save_path):
    """Generates mock data to prevent pipeline blocking."""
    data = pd.DataFrame({
        'admin_unit': ['Unit_' + str(i) for i in range(10)],
        'lat': np.random.uniform(-1.0, 0.0, 10),
        'long': np.random.uniform(34.0, 35.0, 10),
        'population': np.random.randint(50000, 500000, 10),
        'expected_cases': np.random.randint(50, 500, 10),
        'reported_cases': np.random.randint(10, 400, 10)
    })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_csv(save_path, index=False)
    print(f"Generated synthetic test data at: {save_path}")
    return data


def run_surveillance_pipeline(csv_filename):
    input_path = os.path.join("data", "raw", csv_filename)
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"results_{csv_filename}")

    # Check if data exists, if not, generate it
    if not os.path.exists(input_path):
        print(f"Warning: {csv_filename} not found. Generating synthetic data...")
        df = generate_synthetic_data(input_path)
    else:
        df = pd.read_csv(input_path)

    # 1. SENSOR (Graph Engine)
    builder = TopologyAwareGraphBuilder(radius_km=15.0, m0_fraction=0.05)
    graph_tensor = builder.build_graph(df)

    # 2. BRAIN (T-CGNN Model)
    model = TopologyCausalGNN(num_node_features=4)
    model.eval()

    with torch.no_grad():
        predictions = model(graph_tensor.x, graph_tensor.edge_index)

    # 3. OUTPUT
    df['void_risk_score'] = predictions.numpy()
    df.to_csv(output_path, index=False)
    print(f"Pipeline Complete. Results saved to: {output_path}")


if __name__ == "__main__":
    # This will now work even if you don't have a CSV file!
    run_surveillance_pipeline("test_run.csv")