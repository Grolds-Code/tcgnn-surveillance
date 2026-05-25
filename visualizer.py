import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors


class TopologyVisualizer:
    """
    Renders a 2D map of the health facility network, highlighting
    structural surveillance voids identified by the GNN, and saves the output.
    """

    def __init__(self, results_csv, radius_km=15.0):
        self.filepath = os.path.join("data", "processed", results_csv)
        self.radius = radius_km

        # Ensure a folder exists to save the high-res images
        self.output_dir = "visualizations"
        os.makedirs(self.output_dir, exist_ok=True)

    def draw_network(self, save_filename="topology_map.png"):
        print("--- Initializing 2D Topology Render ---")
        if not os.path.exists(self.filepath):
            print(f"Error: Results not found at {self.filepath}. Run main_pipeline.py first.")
            return

        df = pd.read_csv(self.filepath)

        # Create an empty mathematical graph
        G = nx.Graph()

        # 1. Add Facilities (Nodes) to the graph
        for index, row in df.iterrows():
            # If the risk score is > 0.5, we flag it as a void
            is_void = row['void_risk_score'] > 0.5

            G.add_node(
                index,
                pos=(row['long'], row['lat']),
                is_void=is_void,
                risk=row['void_risk_score']
            )

        # 2. Add Connections (Edges) based on your topology rules
        coords = df[['lat', 'long']].values
        # Rough conversion: 1 degree latitude is ~111km
        radius_degrees = self.radius / 111.0

        nn = NearestNeighbors(radius=radius_degrees)
        nn.fit(coords)
        adj_matrix = nn.radius_neighbors_graph(coords, mode='distance')

        # Draw lines between connected facilities
        rows, cols = adj_matrix.nonzero()
        for i, j in zip(rows, cols):
            if i != j:  # Don't connect a facility to itself
                G.add_edge(i, j)

        # 3. Setup the Canvas
        plt.figure(figsize=(10, 8))
        pos = nx.get_node_attributes(G, 'pos')

        # Color coding: Red for Voids, Blue for Normal Facilities
        colors = ['red' if G.nodes[n]['is_void'] else 'dodgerblue' for n in G.nodes()]
        sizes = [300 + (G.nodes[n]['risk'] * 500) for n in G.nodes()]  # Higher risk = bigger circle

        # 4. Render the Graph
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, edgecolors='black')
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')

        # Add labels (Facility IDs)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='white')

        plt.title("Epidemiological Topology Map\n(Red = Structural Void Risk > 50%)", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, linestyle='--', alpha=0.5)

        # 5. Save the Plot (Publication Quality 300 DPI)
        save_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Render complete. Map successfully saved to: {save_path}")

        # 6. Show the Plot on screen (Optional, you can close the window when done)
        plt.show()


if __name__ == "__main__":
    viz = TopologyVisualizer("results_test_run.csv")
    # You can change the filename here for different runs
    viz.draw_network(save_filename="topology_map_v1.png")