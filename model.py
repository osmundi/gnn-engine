
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax

# Custom Message Passing Layer
class CustomMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(CustomMessagePassing, self).__init__(aggr=aggr)
        self.linear = nn.Linear(in_channels, out_channels)
        self.edge_encoder = nn.Linear(15, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
            return self.linear(x_j) + edge_embedding
        return self.linear(x_j)

    def update(self, aggr_out):
        return aggr_out


# Attention-Based Global Pooling
class AttentionGlobalPooling(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(AttentionGlobalPooling, self).__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, batch):
        attention_scores = self.attention_mlp(x).squeeze(-1)  # Shape: [num_nodes_in_batch]
        attention_scores = softmax(attention_scores, batch)  # Softmax per graph

        # Step 2: Weight node features by attention scores
        weighted_x = x * attention_scores.view(-1, 1)  # Shape: [num_nodes_in_batch, in_channels]

        # Step 3: Aggregate weighted node features per graph using global_add_pool
        return global_add_pool(weighted_x, batch)  # Shape: [num_graphs_in_batch, in_channels]


# EPD GNN Architecture for a Single Graph
class AttentionEPDGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_processors=2):
        super(AttentionEPDGNN, self).__init__()

        # Encoder: Node feature transformation
        # Raw node features (like piece type, color, position) are usually sparse or simple.
        # The encoder learns a richer, task-specific representation in the hidden embedding space.
        self.encoder = nn.Linear(in_channels, hidden_channels)

        # Processor: Stack of message-passing layers
        self.processors = nn.ModuleList([
            CustomMessagePassing(hidden_channels, hidden_channels)
            for _ in range(num_processors)
        ])

        # Attention-based global pooling
        self.attention_pooling = AttentionGlobalPooling(
            hidden_channels, hidden_channels // 2)

        # Decoder: Fully connected layers for graph-level output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Encoder: Transform node features
        x = self.encoder(x)
        x = F.relu(x)

        # Processor: Message passing layers
        for processor in self.processors:
            x = processor(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)

        # Attention-based pooling
        graph_embedding = self.attention_pooling(x, batch)

        # Decoder: Predict graph-level output
        out = self.decoder(graph_embedding)
        return out


class GNNModel(nn.Module):
    def __init__(self, num_features=13, hidden_size=64, target_size=1):
        super(GNNModel, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.convs = [GATConv(self.num_features, self.hidden_size, edge_dim = 15).to('cuda'),
                      GATConv(self.hidden_size, self.hidden_size, edge_dim = 15).to('cuda')]
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv in self.convs[:-1]:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x=x, edge_index=edge_index, edge_attr=edge_attr) # edge features here as well

        # Global mean pooling: aggregate node features to get a graph-level embedding
        x = global_mean_pool(x, batch)  # 'batch' tensor tells which nodes belong to which graph

        x = self.linear(x)
        return torch.tanh(x)
