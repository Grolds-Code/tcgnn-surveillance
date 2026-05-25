import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from backend.graph_engine import TopologyAwareGraphBuilder
from backend.tcgnn_model import TopologyCausalGNN


def train_model(csv_filename, epochs=200, learning_rate=0.01):
    """
    Trains the T-CGNN on historical data to learn the 'Geometry of Silence'.
    """
    print(f"--- Initializing Training Protocol for {epochs} Epochs ---")

    # 1. LOAD DATA
    input_path = os.path.join("data", "raw", csv_filename)
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}. Please run main_pipeline.py to generate test data.")
        return

    df = pd.read_csv(input_path)

    # 2. BUILD THE GRAPH (The Environment)
    builder = TopologyAwareGraphBuilder(radius_km=15.0, m0_fraction=0.05)
    graph_tensor = builder.build_graph(df)

    # 3. CREATE 'GROUND TRUTH' LABELS
    # In a real scenario, this would be historical data where a void actually occurred (1) or didn't (0).
    # For this simulation, we will define a "structural void" as any facility where
    # the reported cases are less than 30% of expected cases.
    true_labels = (df['reported_cases'] < (0.3 * df['expected_cases'])).astype(float).values
    y_true = torch.tensor(true_labels, dtype=torch.float).view(-1, 1)

    # 4. INITIALIZE THE BRAIN
    model = TopologyCausalGNN(num_node_features=4, hidden_channels=32)

    # 5. INITIALIZE OPTIMIZER & LOSS FUNCTION
    # Adam is the industry standard optimizer for neural networks
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Binary Cross Entropy Loss (Perfect for predicting Yes/No probabilities like a Void)
    criterion = nn.BCELoss()

    # --- THE TRAINING LOOP ---
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear old gradients

        # Forward Pass: Make a prediction
        out = model(graph_tensor.x, graph_tensor.edge_index)

        # Calculate how wrong the prediction was
        loss = criterion(out, y_true)

        # Backward Pass: Calculate the adjustments needed
        loss.backward()

        # Optimizer Step: Apply the adjustments to the model weights
        optimizer.step()

        # Print progress every 20 epochs
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    # --- SAVE THE TRAINED BRAIN ---
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "tcgnn_trained_weights.pth")
    torch.save(model.state_dict(), save_path)

    print("\n[SUCCESS] Training Complete!")
    print(f"Model weights saved to: {save_path}")


if __name__ == "__main__":
    # Ensure you have run main_pipeline.py at least once to generate 'test_run.csv'
    train_model("test_run.csv", epochs=200, learning_rate=0.01)