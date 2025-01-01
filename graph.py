import sys
import os
from functools import partial

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import chess
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn.models import GAT
from torch_geometric.nn import global_mean_pool
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from dataset import FensOnDisk
from util import *
from fen_parser import FenParser


def create_edges() -> torch.Tensor:

    # Map square to (row, col) and vice versa
    def coords_to_square(row, col):
        return row * 8 + col

    def square_to_coords(square):
        return square // 8, square % 8

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


def fen_to_node_features(fen_parser):
    """
    node features: these represent the pieces on the squares (nodes)

    0: white/black to move (global feature)
    1: current player pawn
    2: current player knight
    3: current player bishop
    4: current player rook
    5: current player queen
    6: current player king
    7: opponent pawn
    ...
    12. opponent king
    13. current player attacks
    14. opponent attacks
    """
    num_nodes = 64
    node_features = torch.zeros((num_nodes, 13))
    white_to_move = fen_parser.white_to_move()

    for i, row in enumerate(list(reversed(fen_parser.parse()))):
        for j, square in enumerate(row):
            if len(square.strip()) > 0:
                node_features[chess.square(j, i)] = fen_parser.piece_to_tensor(
                    square, white_to_move)

            # whose turn it's to move
            # NOTE: this could be encoded also as a global feature for the whole graph
            # but we'll do it like this for simplicity
            if white_to_move:
                node_features[chess.square(j, i)][0] = 1

            attacked_by_white = fen.get_board().is_attacked_by(chess.WHITE, chess.square(j,i))
            if attacked_by_white:
                node_features[chess.square(j, i)][13] = 1
                
            attacked_by_black = fen.get_board().is_attacked_by(chess.BLACK, chess.square(j,i))
            if attacked_by_black:
                node_features[chess.square(j, i)][14] = 1

    return node_features


# edge features: these represent legal moves between squares (nodes)
def alphabetical_distance(char1, char2):
    return abs(ord(char1.lower()) - ord(char2.lower()))


def fen_to_edge_features(fen_parser, edges):

    # initialize edge feature tensor
    edge_features = torch.zeros((edges.size(1), 12))

    white_to_move = fen_parser.white_to_move()

    # 1. map pieces to squares
    pieces = fen_parser.get_board().piece_map()

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
        edge_features[edge][pieces[start].piece_type] = 1
        # if pieces[start].piece_type == 1:
        #     # pawn
        #     edge_features[edge][1] = 1
        # elif pieces[start].piece_type == 2:
        #     # knight
        #     edge_features[edge][2] = 1
        # elif pieces[start].piece_type == 3:
        #     # bishop
        #     edge_features[edge][3] = 1
        # elif pieces[start].piece_type == 4:
        #     # rook
        #     edge_features[edge][4] = 1
        # elif pieces[start].piece_type == 5:
        #     # queen
        #     edge_features[edge][5] = 1
        # elif pieces[start].piece_type == 6:
        #     # king
        #     edge_features[edge][6] = 1
        # else:
        #     assert False, f"Did not recognize piece symbol: {pieces[start].piece_type}"

        # castling
        if move == "e1g1" and fen_parser.can_castle('w', 'k'):
            edge_features[edge][7] = 1
        if move == "e1c1" and fen_parser.can_castle('w', 'q'):
            edge_features[edge][8] = 1
        if move == "e8g8" and fen_parser.can_castle('b', 'k'):
            edge_features[edge][9] = 1
        if move == "e8c8" and fen_parser.can_castle('b', 'q'):
            edge_features[edge][10] = 1

        # pawn promotion
        if promote:
            edge_features[edge][11] = 1
            promote = False

        # edge length
        # edge_features[edge][15] = abs(int(move[1]) - int(move[3]))
        # edge_features[edge][16] = alphabetical_distance(move[0], move[2])

        # print(edge_features[edge])

    return edge_features


# one graph is ~150kb, so creating 13M of these takes about 2TB disk space
# -> create graphs on the fly for now
def create_graph_from_fen(fen_parser, edges) -> Data:
    return Data(
        x=fen_to_node_features(fen_parser),
        edge_index=edges,
        edge_attr=fen_to_edge_features(fen_parser, edges)
    )


def parse_chess_data(chess_data: str) -> list:
    fen, evaluation = chess_data.split(',')
    return FenParser(fen), parse_evaluation(evaluation)


class FENDataset(Dataset):
    # TODO: move to datasets.py

    def __init__(self, fen_file, all_evaluations, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        with open(fen_file, 'r') as f:
            self.fen_list = [line.strip() for line in f]

        self.edges = create_edges()
        self.evaluations = all_evaluations
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
        # graph.y = torch.tensor(evaluation, dtype=torch.float)
        graph.y = normalize_evaluation(evaluation, self.evaluations)

        self.cache[idx] = graph

        return graph


def infer(fen, cp):
    print(f"Evaluate position: {fen}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = instantiate(cfg.model)

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


@hydra.main(version_base=None, config_path="config", config_name="default")
def train(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = load_datasets(cfg)
    # train_loader, val_loader, test_loader = load_datasets(cfg, on_disk=False)

    batch_size = cfg.training.batch_size

    model = instantiate(cfg.model)

    print(model)

    # Move model to GPU if available
    model.to(device)

    # Optimizer
    # lr = 0.001
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Mean Squared Error for regression (centipawn evaluation)
    # CrossEntropyLoss for classification task

    # hack
    if cfg.loss._target_ == "torch.nn.CrossEntropyLoss":
        class_weights = torch.load(
            f"dataset/raw/class_weights/{cfg.training.data.rsplit('.', 1)[0]}.pt")
        print("class weights:", class_weights)
        criterion = instantiate(cfg.loss, weight=class_weights)
    else:
        criterion = instantiate(cfg.loss)

    # Number of epochs to train for
    epochs = 200

    # Set remote MLflow tracking URI
    tracking_url = os.environ['TRACKING_URL']
    tracking_url = False

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

            correct = 0
            total = 0

            for batch in tqdm(train_loader):
                # Move batch to device (GPU or CPU)
                batch = batch.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass: Get model predictions
                output = model(batch)

                # print(output)
                # print(output.shape)
                # print(batch.y)
                # print(batch.y.shape)

                # Calculate the loss, assuming batch.y contains the evaluations
                # loss = criterion(output, batch.y)

                loss = criterion(output, batch.y.long())
                # for MSE
                # loss = criterion(output, batch.y.view(-1, 1))

                # Backpropagation
                loss.backward()

                optimizer.step()

                # Track loss
                train_loss += loss.item()

                # Calculate accuracy

                # Predicted classes: index of the max logit
                # _, predicted = torch.max(output, dim=1)  # Shape: [batch_size]

                # Calculate correct predictions
                # correct += (predicted == batch.y).sum().item()
                # total += batch.y.size(0)

            # Accuracy
            # accuracy = 100 * correct / total
            accuracy = 0

            # Calculate average training loss and accuracy
            avg_train_loss = train_loss / len(train_loader)

            print(
                f'Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Acc: {accuracy:.4f}')

            if tracking_url:
                # Log metrics
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            if epoch % 10 == 0:
                checkpoint = 'checkpoints/100k/AttentionEPDGNN_checkpoint_{:03d}.pt'.format(
                    epoch+1)
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

                    # loss = criterion(output, batch.y.view(-1, 1))
                    loss = criterion(output, batch.y.long())
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
            # loss = criterion(output, batch.y.view(-1, 1))
            loss = criterion(output, batch.y.long())
            test_loss += loss.item()

            # MAE calculation
            # predicted = output.squeeze()
            # mae = torch.mean(torch.abs(predicted - batch.y))
            # total_mae += mae.item()

    # Calculate training accuracy
    # avg_mae = total_mae / len(test_loader)

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
            cur_eval = infer(
                f"{board.fen()},0", 'checkpoints/500k/AttentionEPDGNN_checkpoint_491.pt')

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


def from_fens(fen: str, evaluation: int, edges: torch.Tensor) -> Data:
    fen_parser = FenParser(fen)
    graph = create_graph_from_fen(fen_parser, edges)
    graph.y = evaluation
    return graph


def load_datasets(cfg, on_disk=True):
    """
    Separate data to training, validation and testing sets

    Return dataloaders for these split datasets


    DataBatch(x=[2048, 13], edge_index=[2, 59392], edge_attr=[59392, 15], y=[32], batch=[2048], ptr=[33])

    DataBatch(x=[1024, 13], edge_index=[2, 59392], edge_attr=[29696, 15], y=[32], batch=[1024], ptr=[33])


    """

    batch_size = cfg.training.batch_size
    chess_data = cfg.training.data

    if on_disk:
        edges = create_edges()
        train_set = FensOnDisk('dataset/', chess_data, split='train',
                               from_fens=from_fens, edges=edges)
        val_set = FensOnDisk('dataset/', chess_data, split='val',
                             from_fens=from_fens, edges=edges)
        test_set = FensOnDisk('dataset/', chess_data, split='test',
                              from_fens=from_fens, edges=edges)
    else:
        all_evaluations = normalize_evalutions(chess_data)
        dataset = FENDataset(chess_data, all_evaluations)
        # Define lengths for train, validation, and test
        train_len = int(0.7 * len(dataset))  # 70% for training
        val_len = int(0.15 * len(dataset))   # 15% for validation
        test_len = len(dataset) - train_len - val_len  # 15% for testing

        # Split the dataset into train, validation, and test sets
        train_set, val_set, test_set = random_split(
            dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=2,
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    # train_loader, val_loader, test_loader = load_datasets()
    #
    # for step, batch in enumerate(tqdm(train_loader)):
    #     continue

    # Train batch:
    # DataBatch(x=[2048, 13], edge_index=[2, 59392], edge_attr=[59392, 15], y=[32], batch=[2048], ptr=[33])

    # Normalize evaluations
    # all_evaluations = normalize_evalutions(chess_data)

    # train
    train()

    # game loop
    # play()

    # Evalution: 85
    # Win rate: 29.103874448486668 per cent
    # Bin: 37
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
