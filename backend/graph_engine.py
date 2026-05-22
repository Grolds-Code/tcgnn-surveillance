import torch
from torch_geometric.data import (Data)
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


class TopologyAwareGraphBuilder:
    """
    Constructs a PyTorch Geometric Graph where nodes are health facilities.
    Embeds local topological invariants (Distance-to-Measure) directly into the node features
    to enable Out-of-Distribution (OOD) causal forecasting.
    """

    def __init__(self, radius_km=5.0, m0_fraction=0.05):
        self.radius_km = radius_km
        self.m0_fraction = m0_fraction

    def _latlon_to_cartesian(self, lat, lon):
        """
        Projects WGS84 to a local metric Cartesian plane to prevent
        Euclidean distortion near the equator during topological filtration.
        """
        R = 6371.0  # Earth radius in km
        x = R * np.cos(np.radians(lat)) * np.radians(lon)
        y = R * np.radians(lat)
        return np.column_stack((x, y))

    def _compute_local_dtm(self, coords, k):
        """
        Computes the geometric Distance-to-Measure (DTM) for each node.
        This provides the neural network with a stable topological signature.
        """
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        # Quadratic mean of distances to k nearest neighbors
        dtm = np.sqrt(np.mean(distances ** 2, axis=1))
        return dtm

    def build_graph(self, df: pd.DataFrame) -> Data:
        """
        Ingests tabular facility data and compiles the PyTorch Geometric tensor.
        """
        # 1. Project Coordinates
        coords = self._latlon_to_cartesian(df['lat'].values, df['long'].values)

        # 2. Compute Topological Features (Local DTM)
        n = len(df)
        k = max(2, int(n * self.m0_fraction))
        local_dtm = self._compute_local_dtm(coords, k)

        # 3. Build Node Feature Matrix (X)
        # Features: [Population, Expected Cases, Reported Cases, Local DTM Signature]
        features = np.column_stack((
            df['population'].values,
            df['expected_cases'].values,
            df['reported_cases'].values,
            local_dtm
        ))
        x = torch.tensor(features, dtype=torch.float)

        # 4. Build Edge Index (Adjacency Matrix) based on spatial radius
        nbrs = NearestNeighbors(radius=self.radius_km, algorithm='ball_tree').fit(coords)
        adj_matrix = nbrs.radius_neighbors_graph(coords).tocoo()

        # Remove self-loops (diagonal) for clean GNN message passing
        mask = adj_matrix.row != adj_matrix.col
        row = adj_matrix.row[mask]
        col = adj_matrix.col[mask]

        edge_index = torch.tensor(
            np.vstack((row, col)),
            dtype=torch.long
        )

        # 5. Create PyTorch Geometric Data Object
        graph = Data(x=x, edge_index=edge_index)
        return graph


# ==========================================
# SYSTEM SELF-TEST & VISUALIZATION
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_networkx
    import networkx as nx

    print("Initializing the T-CGNN Graph Engine...")

    # Simulating a 5-unit subset of the Nyanza Basin
    mock_data = pd.DataFrame({
        'admin_unit': ['Kisumu Central', 'Kisumu East', 'Kisumu West', 'Seme', 'Muhoroni'],
        'lat': [-0.0917, -0.0512, -0.1234, -0.0423, -0.1567],
        'long': [34.7617, 34.8312, 34.6934, 34.6512, 35.1823],
        'population': [409000, 178000, 155000, 93000, 176000],
        'expected_cases': [310, 145, 120, 80, 150],
        'reported_cases': [290, 140, 110, 20, 145]  # Note the massive drop in 'Seme'
    })

    builder = TopologyAwareGraphBuilder(radius_km=15.0, m0_fraction=0.05)
    surveillance_graph = builder.build_graph(mock_data)

    print("\n[SUCCESS] Topology-Aware Graph Successfully Compiled!")
    print(f"Number of Facility Nodes: {surveillance_graph.num_nodes}")
    print(f"Number of Connections (Edges): {surveillance_graph.num_edges}")

    # --- VISUALIZATION BLOCK ---
    print("\nGenerating visual plot...")

    # Convert PyTorch Geometric graph to a NetworkX undirected graph
    G = to_networkx(surveillance_graph, to_undirected=True)

    # Map the true geographic coordinates to the graph nodes for accurate plotting
    pos = {i: (mock_data.iloc[i]['long'], mock_data.iloc[i]['lat']) for i in range(len(mock_data))}

    # Extract the DTM signatures to color-code the nodes
    dtm_signatures = surveillance_graph.x[:, 3].numpy()

    plt.figure(figsize=(10, 7))
    plt.title("T-CGNN Facility Network (Nodes colored by Topological DTM Signature)", fontsize=14)

    # Draw the graph network
    nx.draw(
        G, pos,
        with_labels=False,
        node_color=dtm_signatures,
        cmap=plt.cm.plasma,
        node_size=800,
        edge_color="gray",
        alpha=0.8
    )

    # Add custom labels with the Admin Unit names
    labels = {i: mock_data.iloc[i]['admin_unit'] for i in range(len(mock_data))}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()