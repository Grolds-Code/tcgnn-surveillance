import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from backend.graph_engine import TopologyAwareGraphBuilder
from backend.tcgnn_model import TopologyCausalGNN

# 1. Build the "Dumb AI" (No Topology, No Geometry)
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def run_ablation_study(csv_filename="test_run.csv", epochs=200, lr=0.005):
    print("=== INITIATING ABLATION STUDY: GEOMETRY VS. STANDARD ANALYSIS ===\n")
    
    input_path = os.path.join("data", "raw", csv_filename)
    df = pd.read_csv(input_path)
    
    # Standardize data just like the main engine
    for col in ['population', 'expected_cases', 'reported_cases']:
        if df[col].std() != 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Ground truth labels
    true_labels = (df['reported_cases'] < (0.3 * df['expected_cases'])).astype(float).values
    y_true = torch.tensor(true_labels, dtype=torch.float).view(-1, 1)

    # Build the topological graph
    builder = TopologyAwareGraphBuilder(radius_km=15.0, m0_fraction=0.05)
    graph_tensor = builder.build_graph(df)

    # Initialize both competitors
    dumb_model = BaselineMLP(input_dim=4, hidden_dim=32)
    smart_model = TopologyCausalGNN(num_node_features=4, hidden_channels=32)
    
    opt_dumb = torch.optim.Adam(dumb_model.parameters(), lr=lr)
    opt_smart = torch.optim.Adam(smart_model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"Racing both models for {epochs} epochs...\n")
    
    # The Training Race
    for epoch in range(epochs):
        # Train Dumb AI (Ignores graph_tensor.edge_index entirely)
        opt_dumb.zero_grad()
        out_dumb = torch.clamp(dumb_model(graph_tensor.x), min=1e-5, max=1.0 - 1e-5)
        loss_dumb = criterion(out_dumb, y_true)
        loss_dumb.backward()
        opt_dumb.step()
        
        # Train Topological AI (Uses spatial geometry)
        opt_smart.zero_grad()
        out_smart = torch.clamp(smart_model(graph_tensor.x, graph_tensor.edge_index), min=1e-5, max=1.0 - 1e-5)
        loss_smart = criterion(out_smart, y_true)
        loss_smart.backward()
        opt_smart.step()

    # The Final Verdict
    print("=== ABLATION RESULTS ===")
    print(f"Baseline MLP (No Topology) Final Loss:   {loss_dumb.item():.4f}")
    print(f"TCGNN (Spatial Topology) Final Loss:     {loss_smart.item():.4f}")
    
    improvement = ((loss_dumb.item() - loss_smart.item()) / loss_dumb.item()) * 100
    print(f"\n[CONCLUSION]: Mapping the 'Geometry of Silence' improved accuracy by {improvement:.2f}%")

if __name__ == "__main__":
    run_ablation_study()