import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class TopologyCausalGNN(torch.nn.Module):
    """
    The Brain: A Graph Neural Network that learns the structural
    topology of the health system to predict surveillance voids.
    """

    def __init__(self, num_node_features, hidden_channels=32):
        super(TopologyCausalGNN, self).__init__()
        # GraphSAGE layers for topological message passing
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Final linear layer to output a single risk score
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        # First topological pass
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second topological pass
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Output prediction mapping
        x = self.out(x)
        # Sigmoid ensures the risk score is a probability between 0.0 and 1.0
        return torch.sigmoid(x)