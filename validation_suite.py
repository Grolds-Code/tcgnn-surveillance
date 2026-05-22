import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from backend.graph_engine import TopologyAwareGraphBuilder
from backend.tcgnn_model import TopologyCausalGNN
import torch


class ValidationSuite:
    def __init__(self, model, builder):
        self.model = model
        self.builder = builder

    def jaccard_similarity(self, y_true, y_pred):
        """Measures the overlap between actual and predicted voids."""
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        return intersection / (union + 1e-9)

    def backtest(self, historical_df, true_labels):
        """
        Runs the model against known historical data.
        true_labels: binary array (1 = void occurred, 0 = no void)
        """
        print("--- Running Backtest ---")
        graph = self.builder.build_graph(historical_df)

        with torch.no_grad():
            preds = self.model(graph.x, graph.edge_index)
            preds_binary = (preds > 0.5).int().numpy().flatten()

        jaccard = self.jaccard_similarity(true_labels, preds_binary)
        precision = precision_score(true_labels, preds_binary)
        recall = recall_score(true_labels, preds_binary)

        print(f"Jaccard Index: {jaccard:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        return jaccard, precision, recall

    def stability_test(self, df, perturbations=0.1):
        """
        Sensitivity Analysis: Artificially degrades infrastructure to see
        if the model detects the resultant structural void.
        """
        print(f"\n--- Running Sensitivity Analysis ({perturbations * 100}% degradation) ---")
        df_degraded = df.copy()
        # Randomly mask 10% of reported cases to simulate a reporting system failure
        indices_to_mask = np.random.choice(df.index, int(len(df) * perturbations), replace=False)
        df_degraded.loc[indices_to_mask, 'reported_cases'] = 0

        graph = self.builder.build_graph(df_degraded)

        with torch.no_grad():
            preds = self.model(graph.x, graph.edge_index)
            avg_risk = preds.mean().item()

        print(f"Mean Risk Score under stress: {avg_risk:.4f}")
        return avg_risk


if __name__ == "__main__":
    # Initialize components
    builder = TopologyAwareGraphBuilder(radius_km=15.0)
    model = TopologyCausalGNN(num_node_features=4)
    model.eval()

    suite = ValidationSuite(model, builder)

    # Generate dummy validation data
    data = pd.DataFrame({
        'lat': np.random.uniform(-1, 0, 50),
        'long': np.random.uniform(34, 35, 50),
        'population': np.random.randint(1000, 10000, 50),
        'expected_cases': np.random.randint(10, 100, 50),
        'reported_cases': np.random.randint(5, 100, 50)
    })

    # 1. Backtest vs Dummy Truth
    true_outcomes = np.random.randint(0, 2, 50)
    suite.backtest(data, true_outcomes)

    # 2. Stability Test
    suite.stability_test(data)