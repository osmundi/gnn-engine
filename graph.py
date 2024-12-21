import sys
import os
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
import torch.nn.functional as F
import torch.nn as nn
import gc
from torch_geometric.data import Dataset, InMemoryDataset
import re
from itertools import chain
import chess
from torch_geometric.data import Data
from functools import partial
import torch
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch


# Map square to (row, col) and vice versa
def square_to_coords(square):
    return square // 8, square % 8


def coords_to_square(row, col):
    return row * 8 + col


def create_edges() -> torch.Tensor:

    edge_index = []

    def add_edges(moves, square, row, col):
        for dr, dc in moves:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                target_square = coords_to_square(nr, nc)
                edge_index.append([square, target_square])

    # Total nodes (squares)
    num_nodes = 64

    # Directions for knight moves
    knight_moves = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]

    # Directions for queen moves (also covers other piece moves)
    queen_moves = [
        (dx, dy)
        for dx in range(-7, 8) for dy in range(-7, 8)
        if (dx == 0) != (dy == 0) or abs(dx) == abs(dy)
    ]

    # Add edges for each square
    for square in range(num_nodes):
        row, col = square_to_coords(square)
        add_edges_for_square = partial(
            add_edges, square=square, col=col, row=row)
        add_edges_for_square(knight_moves)
        add_edges_for_square(queen_moves)

    # Convert edge list to tensor
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


class FenParser():
    def __init__(self, fen_str):
        self.fen_str = fen_str

    def parse(self):
        ranks = self.fen_str.split(" ")[0].split("/")
        pieces_on_all_ranks = [self.parse_rank(rank) for rank in ranks]
        return pieces_on_all_ranks

    def parse_rank(self, rank):
        rank_re = re.compile("(\\d|[kqbnrpKQBNRP])")
        piece_tokens = rank_re.findall(rank)
        pieces = self.flatten(map(self.expand_or_noop, piece_tokens))
        return pieces

    def flatten(self, lst):
        return list(chain(*lst))

    def expand_or_noop(self, piece_str):
        piece_re = re.compile("([kqbnrpKQBNRP])")
        retval = ""
        if piece_re.match(piece_str):
            retval = piece_str
        else:
            retval = self.expand(piece_str)

        return retval

    def expand(self, num_str):
        return int(num_str)*" "

    def white_to_move(self):
        return True if self.fen_str.split()[1] == 'w' else False

    def can_castle(self, color, side):
        castling_rights = self.fen_str.split()[2]

        if castling_rights == '-':
            return False

        if color == 'w':
            if side == 'k':
                return True if 'K' in castling_rights else False
            elif side == 'q':
                return True if 'Q' in castling_rights else False

        if color == 'b':
            if side == 'k':
                return True if 'k' in castling_rights else False
            elif side == 'q':
                return True if 'q' in castling_rights else False

        return False

    def get_board(self):
        return chess.Board(self.fen_str)

    def legal_moves(self):
        return self.get_board().legal_moves


def calculate_inverse_output(input_value: str) -> int:
    """
    Define the range of the output values checkmate in 100 moves is equivalent 
    to +10 evaluation and mate in 1 move is equivalent to +75 evaluation
    """
    max_output = 99_000
    min_output = 1_000

    # Calculate the output using an inverse proportional formula
    output = max_output - (input_value * (max_output - min_output) / 99)
    # Clamp output within range
    return int(max(min_output, min(max_output, output)))


# there are evaluations in the range 1000 - 9999 which are unnecessary big -> might be good idea to normalize them
def normalize_log_evaluation(evaluation: int) -> float:
    """
    For centipawn evaluations, signed log transformation is often a good choice because:

    It compresses large values.
    It keeps smaller values distinguishable.
    It ensures a smooth distribution of data for regression.

    Why Normalize?

    1. Gradient Stability: Large differences in target values can lead to large gradients during backpropagation, which may destabilize training or cause slower convergence.
    2. Model Sensitivity: If the majority of your data lies in the 0–100 range but you also have outliers in the 1000–9999 range, the model might overfit or focus disproportionately on these outliers.
    3. Consistency: Normalizing the data ensures all values are on a similar scale, making the model's predictions more interpretable.
    """
    data = torch.tensor([evaluation], dtype=torch.float)
    return torch.sign(data) * torch.log(torch.abs(data) + 1)

def normalize_evaluation(evaluation: int) -> float:
    evaluations = all_evaluations

    # Compute median and IQR
    median = torch.median(evaluations)
    q1 = torch.quantile(evaluations, 0.25)
    q3 = torch.quantile(evaluations, 0.75)
    iqr = q3 - q1

    # Robust scaling
    robust_scaled_evaluations = (evaluation - median) / iqr

    # Optionally, scale robust outputs to [-1, 1] by dividing further
    return torch.tanh(robust_scaled_evaluations)  # Limits to [-1, 1]

def denormalize_evaluation(normalized_data: float) -> float:
    evaluations = all_evaluations

    # Compute median and IQR
    median = torch.median(evaluations)
    q1 = torch.quantile(evaluations, 0.25)
    q3 = torch.quantile(evaluations, 0.75)
    iqr = q3 - q1

    evaluation = torch.atanh(normalized_data)  # Inverse of tanh

    # Robust scaling
    return (evaluation * iqr + median)


def denormalize_log_evaluation(normalized_data: float) -> float:
    return np.sign(normalized_data) * (np.expm1(np.abs(normalized_data)))


def win_rate_params(position):
    """
    Calculate win rate parameters based on a position.

    Parameters:
        position (dict): A dictionary containing counts of pieces.
            Expected keys: 'PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN'.

    Returns:
        tuple: A tuple (a, b) representing win rate parameters.
    """
    # Calculate material count
    material = (
        position.get('PAWN', 0)
        + 3 * position.get('KNIGHT', 0)
        + 3 * position.get('BISHOP', 0)
        + 5 * position.get('ROOK', 0)
        + 9 * position.get('QUEEN', 0)
    )

    # Clamp material count and normalize
    m = np.clip(material, 17, 78) / 58.0

    # Coefficients for the polynomial model
    as_c = [-37.45051876, 121.19101539, -132.78783573, 420.70576692]
    bs_c = [90.26261072, -137.26549898, 71.10130540, 51.35259597]

    a = as_c[0] * m**3 + as_c[1] * m**2 + as_c[2] * m + as_c[3];
    b = bs_c[0] * m**3 + bs_c[1] * m**2 + bs_c[2] * m + bs_c[3];

    return a, b

def win_rate_model(eval_score, position):
    """
    Calculate the win rate model based on the evaluation and position.

    Translated from stockfish c++ code:
    https://github.com/official-stockfish/Stockfish/blob/cf10644d6e2592e663e48b3d41dae07e7294166e/src/uci.cpp#L527

    Win rate probabilities as a function of evalution:
    https://user-images.githubusercontent.com/4202567/206894542-a5039063-09ff-4f4d-9bad-6e850588cac9.png

    Parameters:
        eval_score (float): The evaluation score.
        position (dict): A dictionary containing counts of pieces.
            Expected keys: 'PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN'.

    Returns:
        float: Win rate between [0, 1]
    """
    a, b = win_rate_params(position)

    # Normalize eval_score
    v = a * eval_score / 100

    # Calculate win rate using the logistic model
    win_rate = 1 / (1 + np.exp((a - v) / b))

    return win_rate


def win_rate_to_bin(rate, bins=128):
    return int(rate * bins)


def parse_evaluation(evaluation: str) -> int:
    # in centipawns
    evaluation = evaluation.strip()

    # evaluate checkmates (e.g. #+63 / #-0)
    if evaluation.startswith('#'):
        evaluation = evaluation.lstrip('#+')

        if evaluation.startswith('-'):
            black_to_move = True
        else:
            black_to_move = False

        evaluation = abs(int(evaluation))

        if evaluation == 0:
            evaluation = 10_000
        else:
            evaluation = calculate_inverse_output(evaluation)

        return -1*evaluation if black_to_move else evaluation

    return int(evaluation)


def parse_chess_data(chess_data: str) -> list:
    fen, evaluation = chess_data.split(',')
    return FenParser(fen), parse_evaluation(evaluation)


# node features: these represent the pieces on the squares (nodes)
def piece_to_tensor(piece: str) -> torch.tensor:
    piece = chess.Piece.from_symbol(piece)
    node_feature = torch.zeros(13)
    node_feature[piece.__hash__() + 1] = 1
    return node_feature


def fen_to_node_features(fen):
    num_nodes = 64
    node_features = torch.zeros((num_nodes, 13))

    for i, row in enumerate(list(reversed(fen.parse()))):
        for j, square in enumerate(row):
            if len(square.strip()) > 0:
                node_features[chess.square(j, i)] = piece_to_tensor(square)

            # whose turn it's to move
            # NOTE: this could be encoded also as a global feature for the whole graph
            # but we'll do it like this for simplicity i
            if fen.white_to_move():
                node_features[chess.square(j, i)][0] = 1

    return node_features


# edge features: these represent legal moves between squares (nodes)
def alphabetical_distance(char1, char2):
    return abs(ord(char1.lower()) - ord(char2.lower()))


def fen_to_edge_features(fen_parser, edges):

    # initialize edge feature tensor
    edge_features = torch.zeros((edges.size(1), 15))

    white_to_move = fen_parser.white_to_move()

    # 1. map pieces to squares
    pieces = {}
    for i, row in enumerate(list(reversed(fen_parser.parse()))):
        for j, square in enumerate(row):
            if len(square.strip()) > 0:
                pieces[chess.square(j, i)] = chess.Piece.from_symbol(square)

    moves = 0
    promoted = []
    promote = False

    # print(pieces)

    for move in fen_parser.legal_moves():
        # move = chess.Move(start.__hash__(), end.__hash__(), pieces[start].piece_type)
        moves += 1
        move = str(move)
        # print(move)

        # pawn promotions looks like this: a7b8Q (only moves where length > 4)
        # there is alway 4 possible promotions but we are interested only from
        # queen promotions for now, so we skip the rest of the possible promotions
        if len(move) > 4:
            move = move[:4]
            if move in promoted:
                continue
            promoted.append(move)
            promote = True

        start, end = chess.parse_square(move[:2]), chess.parse_square(move[2:])
        # print(start)
        # print(end)

        # calculate correct edge index from start/end
        value_pair = torch.tensor([start, end])
        edge = torch.where((edges[0] == value_pair[0]) & (
            edges[1] == value_pair[1]))[0].item()

        # legal move
        edge_features[edge][0] = 1


        # what piece could move between two nodes
        if pieces[start].piece_type == 1:
            # pawn
            if white_to_move:
                edge_features[edge][1] = 1
            else:
                edge_features[edge][2] = 1
        elif pieces[start].piece_type == 6:
            # king
            if white_to_move:
                edge_features[edge][3] = 1
            else:
                edge_features[edge][4] = 1
        elif pieces[start].piece_type == 5:
            # queen
            edge_features[edge][5] = 1
        elif pieces[start].piece_type == 2:
            # knight
            edge_features[edge][6] = 1
        elif pieces[start].piece_type == 3:
            # bishop
            edge_features[edge][7] = 1
        elif pieces[start].piece_type == 4:
            # rook
            edge_features[edge][8] = 1
        else:
            assert False, f"Did not recognize piece symbol: {pieces[start].piece_type}"

        # castling
        if move == "e1g1" and fen_parser.can_castle('w', 'k'):
            edge_features[edge][10] = 1
        if move == "e1c1" and fen_parser.can_castle('w', 'q'):
            edge_features[edge][11] = 1
        if move == "e8g8" and fen_parser.can_castle('b', 'k'):
            edge_features[edge][12] = 1
        if move == "e8c8" and fen_parser.can_castle('b', 'q'):
            edge_features[edge][13] = 1

        # pawn promotion
        if promote:
            edge_features[edge][14] = 1
            promote = False

        # edge length
        #edge_features[edge][15] = abs(int(move[1]) - int(move[3]))
        #edge_features[edge][16] = alphabetical_distance(move[0], move[2])

        # print(edge_features[edge])

    return edge_features


# one graph is ~150kb, so creating 13M of these takes about 2TB disk space
# -> create graphs on the fly for now
def create_graph_from_fen(fen_parser: str, edges) -> Data:
    return Data(
        x=fen_to_node_features(fen_parser),
        edge_index=edges,
        edge_attr=fen_to_edge_features(fen_parser, edges)
    )


class FENDataset(Dataset):
    def __init__(self, fen_file, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        with open(fen_file, 'r') as f:
            self.fen_list = [line.strip() for line in f]

        self.edges = create_edges()
        self.cache = {}

    def len(self):
        return len(self.fen_list)

    def get(self, idx):

        if idx in self.cache:
            return self.cache[idx]

        fen_str = self.fen_list[idx]
        fen_parser, evaluation = parse_chess_data(fen_str)
        graph = create_graph_from_fen(fen_parser, self.edges)

        # store the evaluation
        #graph.y = torch.tensor(evaluation, dtype=torch.float)
        graph.y = normalize_evaluation(evaluation)

        self.cache[idx] = graph

        return graph


class FENDatasetPreLoaded(InMemoryDataset):
    """FENDataset for preprocessed graphs"""

    def __init__(self, root, fen_file, edges, transform=None, pre_transform=None):
        self.fen_file = fen_file
        self.edges = edges
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read the FEN file and preprocess the data
        with open(self.fen_file, 'r') as f:
            fen_list = [line.strip() for line in f]

        data_list = []
        for fen_str in fen_list:
            fen_parser, evaluation = parse_chess_data(fen_str)
            graph = create_graph_from_fen(fen_parser, self.edges)
            graph.y = normalize_evaluation(evaluation)
            data_list.append(graph)

        # Save preprocessed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


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


class GNNModel(torch.nn.Module):
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


def infer(fen, cp):

    print(f"Evaluate position: {fen}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    in_channels = 13
    hidden_channels = 64
    num_iterations = 3

    # Model
    model = AttentionEPDGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=1
    )

    model.to(device)

    # Load the model weights
    checkpoint = torch.load(cp)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    fen_parser, _ = parse_chess_data(fen)
    edges = create_edges()
    graph = create_graph_from_fen(fen_parser, edges)

    # NOTE: this is just a placeholder (which model assumes to be there)
    batch = torch.zeros(64, dtype=torch.long)
    graph.batch = batch
    graph = graph.to(device)

    with torch.no_grad():
        evaluation = model(graph)
        evaluation = torch.tensor(evaluation.item(), dtype=torch.float)
        return denormalize_evaluation(evaluation)


def train(data):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = FENDataset(data)

    # Separate data to training, validation and testing sets
    # Define lengths for train, validation, and test
    train_len = int(0.7 * len(dataset))  # 70% for training
    val_len = int(0.15 * len(dataset))   # 15% for validation
    test_len = len(dataset) - train_len - val_len  # 15% for testing

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len])

    batch_size = 32

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop

    # Hyperparameters
    in_channels = 13   # Features per node
    hidden_channels = 64
    num_iterations = 3 # message passing rounds

    lr = 0.001

    # Model
    model = AttentionEPDGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=1
    )
    #model = GNNModel()

    # Move model to GPU if available
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Mean Squared Error for regression (centipawn evaluation)
    criterion = torch.nn.MSELoss()

    # Number of epochs to train for
    epochs = 200

    # Set remote MLflow tracking URI
    tracking_url = os.environ['TRACKING_URL']
    #tracking_url = False

    if tracking_url:
        mlflow.set_tracking_uri(tracking_url)

        # Start an MLflow experiment
        mlflow.set_experiment("GNN evaluation training")

    with mlflow.start_run() as run:
        # Log hyperparameters
        if tracking_url:
            mlflow.log_param("num_epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", lr)

        # Training loop
        for epoch in range(epochs):
            model.train()  # Set model to training mode
            train_loss = 0.0
            total_mae = 0

            for batch_id, batch in enumerate(train_loader):
                # Move batch to device (GPU or CPU)
                batch = batch.to(device)  

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass: Get model predictions
                output = model(batch)

                # Calculate the loss, assuming batch.y contains the evaluations
                #loss = criterion(output, batch.y)
                loss = criterion(output, batch.y.view(-1, 1))

                # Backpropagation
                loss.backward()  

                optimizer.step()

                # Track loss
                train_loss += loss.item()

                # debugging model outputs
                #if batch_id % 100 == 0:
                    #print("-----")
                    #print(output)
                    #print(predicted)
                    #print(batch.y)
                    #print("-----")

            # Calculate average training loss and accuracy
            avg_train_loss = train_loss / len(train_loader)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}')

            if tracking_url:
                # Log metrics
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            if epoch % 10 == 0:
                checkpoint = 'checkpoints/500k/AttentionEPDGNN_checkpoint_{:03d}.pt'.format(epoch+1)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_train_loss,
                }, checkpoint)
                print(f"Checkpoint saved to: {checkpoint}")

            # Validation step (no gradients needed)
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0

            with torch.no_grad():  # No gradients needed during validation
                for batch in val_loader:
                    batch = batch.to(device)

                    output = model(batch)

                    #loss = criterion(output, batch.y)
                    loss = criterion(output, batch.y.view(-1, 1))
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            if tracking_url:
                # Log metrics
                mlflow.log_metric("validation_loss", avg_val_loss, step=epoch)

            print(f'Validation Loss: {avg_val_loss:.4f}')

    # Save the model and log it to the MLflow server
    # mlflow.log_artifact('AttentionEPDGNN_checkpoint.pt')
    if tracking_url:
        mlflow.pytorch.log_model(model, "model")

    # Test after training is done
    model.eval()  # Set model to evaluation mode for testing
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch.y.view(-1, 1))
            test_loss += loss.item()

            # MAE calculation
            #predicted = output.squeeze()
            #mae = torch.mean(torch.abs(predicted - batch.y))
            #total_mae += mae.item()

    # Calculate training accuracy
    #avg_mae = total_mae / len(test_loader)

    avg_test_loss = test_loss / len(test_loader)

    print(f'Test Loss: {avg_test_loss:.4f}')


def play():
    """play a game of chess against computer"""

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 0"
    board = chess.Board(fen)

    while True:

        print(fen)

        best_eval = None
        best_move = None

        for move in board.legal_moves:
            board.push(move)
            cur_eval = infer(f"{board.fen()},0", 'checkpoints/500k/AttentionEPDGNN_checkpoint_491.pt')

            if not best_eval:
                best_eval = cur_eval
                best_move = move
            elif fen.split()[1] == 'w':
                # white to move -> bigger the better
                if best_eval < cur_eval:
                    best_eval = cur_eval
                    best_move = move
            elif fen.split()[1] == 'b':
                # black to move 
                if best_eval > cur_eval:
                    best_eval = cur_eval
                    best_move = move

            board.set_fen(fen)
        
        print(f"Computer move: {best_move} (eval: {best_eval})")
        board.push(best_move)
        print("Your move:")
        human_move = input()
        human_move = chess.Move.from_uci(human_move)

        if human_move in board.legal_moves:
            board.push(human_move)
        else:
            print("Try again:")
            human_move = input()
            board.push(human_move)

        fen = board.fen()



def normalize_evalutions(chess_data):

    data = pd.read_csv(chess_data, header=None, names=["fen", "evaluation"])

    # Assuming "evaluation" is the second column after splitting
    data['evaluation'] = data['evaluation'].apply(
        lambda x: 1000 if '#+' in x else (-1000 if '#-' in x else int(x.strip()))
    )

    # Convert evaluations to a PyTorch tensor
    global all_evaluations
    all_evaluations = torch.tensor(data['evaluation'].values, dtype=torch.float)

if __name__ == '__main__':

    chess_data = "first_10k_evaluations.csv"

    # Normalize evaluations
    # normalize_evalutions(chess_data)

    # train
    # train(chess_data)

    # game loop
    # play()

    position = {'PAWN': 8, 'KNIGHT': 2, 'BISHOP': 2, 'ROOK': 2, 'QUEEN': 0}
    eval_score = 75
    win_rate = win_rate_model(eval_score, position)
    print(f"Evalution: {eval_score}\nWin rate: {win_rate * 100} per cent\n")
    print(f"Bin: {win_rate_to_bin(win_rate)}")
    print("------")

    position = {'PAWN': 8, 'KNIGHT': 2, 'BISHOP': 2, 'ROOK': 2, 'QUEEN': 0}
    eval_score = 85
    win_rate = win_rate_model(eval_score, position)
    print(f"Evalution: {eval_score}\nWin rate: {win_rate * 100} per cent\n")
    print(f"Bin: {win_rate_to_bin(win_rate)}")
    print("------")

    position = {'PAWN': 8, 'KNIGHT': 2, 'BISHOP': 2, 'ROOK': 2, 'QUEEN': 0}
    eval_score = 95
    win_rate = win_rate_model(eval_score, position)
    print(f"Evalution: {eval_score}\nWin rate: {win_rate * 100} per cent\n")
    print(f"Bin: {win_rate_to_bin(win_rate)}")
    print("------")
    
    position = {'PAWN': 8, 'KNIGHT': 2, 'BISHOP': 2, 'ROOK': 2, 'QUEEN': 0}
    eval_score = 110
    win_rate = win_rate_model(eval_score, position)
    print(f"Evalution: {eval_score}\nWin rate: {win_rate * 100} per cent\n")
    print(f"Bin: {win_rate_to_bin(win_rate)}")
    print("------")
