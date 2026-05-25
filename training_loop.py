import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from backend.graph_engine import TopologyAwareGraphBuilder
from backend.tcgnn_model import TopologyCausalGNN

def train_model(csv_filename, epochs=200, learning_rate=0.005):
    print(f"--- Initializing Training Protocol for {epochs} Epochs ---")
    
    input_path = os.path.join("data", "raw", csv_filename)
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}.")
        return

    df = pd.read_csv(input_path)
    
    # 1. Establish the 'Ground Truth' labels FIRST, before we alter the data
    true_labels = (df['reported_cases'] < (0.3 * df['expected_cases'])).astype(float).values
    y_true = torch.tensor(true_labels, dtype=torch.float).view(-1, 1)

    # 2. Biostatistical Rigor: Z-Score Standardization
    # We convert massive integers into standard deviations so the GNN can digest them cleanly
    for col in ['population', 'expected_cases', 'reported_cases']:
        if df[col].std() != 0: # Prevent division by zero
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    # 3. Build the Topological Graph
    builder = TopologyAwareGraphBuilder(radius_km=15.0, m0_fraction=0.05)
    graph_tensor = builder.build_graph(df)

    # 4. Initialize the Brain
    model = TopologyCausalGNN(num_node_features=4, hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss() 

    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_tensor.x, graph_tensor.edge_index)
        
        # 5. The Mathematical Clamp
        # Forces predictions to stay between 0.00001 and 0.99999.
        # This prevents the exact 50.0000 / 30.0000 flatline crash!
        out = torch.clamp(out, min=1e-5, max=1.0 - 1e-5)
        
        loss = criterion(out, y_true)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "tcgnn_trained_weights.pth")
    torch.save(model.state_dict(), save_path)
    
    print("\n[SUCCESS] Training Complete!")

if __name__ == "__main__":  
    train_model("test_run.csv", epochs=200, learning_rate=0.005)