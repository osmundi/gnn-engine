# GNN-Engine: Chess Position Evaluation using Graph Neural Networks

## Project Overview

This project uses Graph Neural Networks (GNNs) to evaluate chess positions. It represents chess boards as graphs where:
- **Nodes** represent the 64 squares on a chessboard
- **Edges** represent possible piece movements (knight moves, queen moves, etc.)
- **Node features** encode piece types, positions, and attack patterns
- **Edge features** encode legal moves, piece types, castling rights, and promotions

The model learns to predict position evaluations (in centipawns or win-rate bins) from chess positions in FEN notation.

## Repository Structure

```
/home/user/gnn-engine/
├── config/
│   └── default.yaml          # Hydra configuration for data, model, training, loss
├── dataset.py                # OnDiskDataset implementation for loading preprocessed graphs
├── fen_parser.py             # FEN string parsing and chess position features
├── graph.py                  # Main training/inference script with graph creation
├── model.py                  # GNN model architectures (GATv2, GAT, AttentionEPDGNN)
├── preprocess.py             # Data preprocessing: evaluation → class bins
├── util.py                   # Utility functions for normalization, win-rate calculations
└── graph.ipynb               # Jupyter notebook for experimentation
```

## Core Components

### 1. FEN Parser (`fen_parser.py`)

**Purpose**: Parse FEN (Forsyth-Edwards Notation) strings to extract chess position information.

**Key Methods**:
- `parse()`: Convert FEN to 8x8 board representation
- `white_to_move()`: Determine whose turn it is
- `can_castle()`: Check castling rights
- `legal_moves()`: Get all legal moves
- `piece_counts()`: Count material on board
- `piece_to_tensor()`: Convert piece symbol to 15-dimensional feature vector

**Node Features (15 dimensions)**:
- `[0]`: White/black to move indicator
- `[1-6]`: Current player pieces (pawn, knight, bishop, rook, queen, king)
- `[7-12]`: Opponent pieces
- `[13]`: Current player attacks this square
- `[14]`: Opponent attacks this square

### 2. Graph Construction (`graph.py`)

**Edge Creation**:
- `create_edges()`: Creates a static edge structure for all possible piece movements
  - Knight moves: L-shaped (8 directions)
  - Queen moves: All ranks, files, and diagonals
  - Returns shape: `[2, num_edges]` tensor

**Node Features**:
- `fen_to_node_features()`: Creates 64x15 tensor representing all squares
  - Encodes piece positions, whose turn, attack patterns

**Edge Features (12 dimensions)**:
- `[0]`: Is this a legal move?
- `[1-6]`: Which piece type can move (pawn, knight, bishop, rook, queen, king)
- `[7-10]`: Castling rights (white kingside, white queenside, black kingside, black queenside)
- `[11]`: Pawn promotion

**Graph Creation**:
- `create_graph_from_fen()`: Combines node features, edges, edge features into PyTorch Geometric `Data` object

### 3. Models (`model.py`)

#### GATv2 (Graph Attention Network v2) - Default Model
```python
GATv2(in_channels=15, hidden_channels=128, out_channels=16, num_layers=4)
```
- Uses edge features (12-dimensional)
- Multiple GATConv layers with ReLU activation
- Global mean pooling for graph-level representation
- Linear classifier for final prediction

#### AttentionEPDGNN (Encode-Process-Decode Architecture)
- **Encoder**: Transform raw features to hidden space
- **Processor**: Multiple message-passing layers
- **Attention Pooling**: Weighted aggregation of node features
- **Decoder**: MLP for graph-level output

#### GAT (Graph Attention Network)
- Similar to GATv2 but with different layer configuration
- Uses edge features throughout

### 4. Dataset (`dataset.py`)

**FensOnDisk**:
- Extends `OnDiskDataset` for efficient on-disk storage
- Loads graphs dynamically during training
- Uses SQLite backend for indexing
- Supports train/val/test splits

**Schema**:
```python
{
    'x': (num_nodes=64, 15),           # Node features
    'edge_index': (2, num_edges),       # Edge connectivity
    'edge_attr': (num_edges, 12),       # Edge features
    'y': int                            # Target evaluation class
}
```

### 5. Preprocessing (`preprocess.py`)

**Purpose**: Convert raw CSV data (FEN, evaluation) into class-based targets.

**Pipeline**:
1. Read CSV with FEN strings and evaluations (centipawns or mate notation)
2. Convert evaluations to win-rate probabilities using Stockfish's model
3. Bin win rates into classes (default: 16 bins)
   - Bins 0-7: White advantage (0-100% win rate)
   - Bins 8-15: Black advantage (0-100% win rate)
4. Compute class weights for balanced training
5. Create train/val/test split indices (90%/5%/5%)

**Evaluation Formats**:
- Centipawns: `85`, `-120` (positive = white advantage)
- Mate: `#+5` (white mates in 5), `#-3` (black mates in 3)

### 6. Utilities (`util.py`)

**Win Rate Model**:
- `win_rate_model()`: Stockfish's logistic win-rate function
  - Considers both evaluation and material count
  - Returns probability in [0, 1]
- `win_rate_to_bin()`: Convert probability to discrete class

**Normalization**:
- `normalize_evaluation()`: Robust scaling using IQR + tanh
- `denormalize_evaluation()`: Inverse transformation
- `normalize_log_evaluation()`: Signed log transform for large values

## Configuration Management

Uses **Hydra** for hierarchical configuration (`config/default.yaml`):

```yaml
data:
  bins: 16                    # Number of output classes
  edge.features: 12
  node.features: 15

training:
  root_dir: /data/
  data: chessData.csv
  batch_size: 32

model:
  _target_: model.GATv2      # Hydra instantiation target
  in_channels: ${data.node.features}
  hidden_channels: 128
  out_channels: ${data.bins}
  num_layers: 4

loss:
  _target_: torch.nn.CrossEntropyLoss
```

**Key Features**:
- Variable interpolation: `${data.node.features}`
- Easy model swapping by changing `_target_`
- Supports both classification (CrossEntropyLoss) and regression (MSELoss)

## Development Workflows

### Training Workflow

1. **Preprocess data** (one-time):
   ```bash
   python preprocess.py
   ```
   - Converts raw CSV to class-based dataset
   - Creates train/val/test splits
   - Computes class weights
   - Saves to `dataset/raw/`

2. **Train model**:
   ```bash
   python graph.py
   ```
   - Loads configuration from `config/default.yaml`
   - Creates dataloaders with OnDiskDataset
   - Trains for 200 epochs with Adam optimizer
   - Saves checkpoints every 10 epochs to `checkpoints/`
   - Logs metrics to MLflow (if configured)

3. **Monitor training**:
   - Training loss and accuracy per epoch
   - Validation loss after each epoch
   - Test loss after training completes

### Inference Workflow

```python
from graph import infer

# Evaluate a position
fen = "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"
checkpoint_path = "checkpoints/100k/AttentionEPDGNN_checkpoint_491.pt"
evaluation = infer(f"{fen},0", checkpoint_path)
print(f"Evaluation: {evaluation} centipawns")
```

### Playing Against the Model

The `play()` function implements a simple game loop:
1. Display current position
2. Model evaluates all legal moves
3. Selects best move based on evaluation
4. Prompts human for move
5. Updates board and repeats

## Key Conventions

### Code Style

1. **PyTorch Geometric Data Format**:
   - Always include `x`, `edge_index`, `edge_attr` in graphs
   - Add `batch` tensor for batching multiple graphs

2. **Device Handling**:
   - Models hardcoded to CUDA in some places (see `model.py:126-127`)
   - Consider removing `.to('cuda')` from model definition

3. **Feature Encoding**:
   - One-hot encoding for piece types
   - Binary indicators for attacks, castling, legal moves
   - Perspective-relative encoding (current player vs opponent)

4. **Evaluation Conventions**:
   - Positive = white advantage, negative = black advantage
   - Checkmate values mapped to large numbers (1000-99000 centipawns)
   - Win rates binned symmetrically for white/black

### Git Workflow

- Current branch: `claude/claude-md-mief1rtjku480s0c-01T8NwAiUnMoXRMYQtdrL8cN`
- Recent commits focus on feature engineering and configuration
- Use descriptive commit messages (e.g., "Add features to empty squares")

### Data Expectations

**Input CSV Format**:
```csv
fen,evaluation
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,25
r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3,-15
```

**Processed Dataset Structure**:
```
dataset/
└── raw/
    ├── chessData.csv                        # Processed: FEN, class label
    ├── unprocessed/chessData.csv            # Original: FEN, evaluation
    ├── ranges/chessData.pt                  # Train/val/test indices
    └── class_weights/chessData.pt           # Balanced class weights
```

## Architecture Decisions

### Why Graph Neural Networks?

1. **Natural Representation**: Chess pieces and their relationships form a natural graph
2. **Permutation Invariance**: Board symmetries handled by graph structure
3. **Relational Reasoning**: GNNs learn how pieces interact through message passing
4. **Attention Mechanisms**: GATv2 learns which squares/pieces are most important

### Why Classification Instead of Regression?

Recent commits show shift from regression (MSE) to classification (CrossEntropyLoss):
- **Better convergence**: Discrete bins easier to learn than continuous values
- **Probabilistic interpretation**: Win-rate bins have clear semantic meaning
- **Balanced training**: Class weights handle imbalanced data

### Edge Construction Strategy

Static edge structure (all possible moves) vs dynamic (legal moves only):
- **Static edges** with legal move indicators as features
- Allows model to learn move patterns across positions
- More edges but consistent graph structure for batching

## Common Tasks for AI Assistants

### Adding New Features

1. **Node features**: Modify `fen_to_node_features()` in `graph.py`
   - Update dimension in `config/default.yaml`: `data.node.features`
   - Update schema in `dataset.py`

2. **Edge features**: Modify `fen_to_edge_features()` in `graph.py`
   - Update dimension in `config/default.yaml`: `data.edge.features`
   - Update schema in `dataset.py`
   - Update edge encoder in `model.py` if needed

### Changing Model Architecture

1. Edit `model.py` to add new model class
2. Update `config/default.yaml`: `model._target_`
3. Ensure model accepts `in_channels`, `out_channels` parameters

### Debugging Training Issues

**Check**:
1. Data loading: Verify dataset paths in config
2. Device placement: Ensure consistent CPU/CUDA usage
3. Loss function: CrossEntropyLoss expects long targets, MSE expects float
4. Class weights: Check if loaded correctly for imbalanced data
5. Batch dimensions: Print batch shapes to verify DataLoader

### Performance Optimization

**Current bottlenecks**:
1. Edge feature computation is slow (many legal moves)
2. Dataset preprocessing not parallelized (marked TODO in `dataset.py:71`)
3. Checkpointing every 10 epochs (can adjust frequency)

**Optimization opportunities**:
- Parallelize preprocessing with multiprocessing
- Cache graph structures (already implemented in `FENDataset`)
- Use OnDiskDataset for large datasets (already implemented)
- Adjust number of DataLoader workers (currently 6)

## Testing and Validation

### Manual Position Testing

```python
# Test specific position
fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
result = infer(f"{fen},0", "checkpoints/model.pt")
```

### Validation Metrics

- **Training Loss**: CrossEntropyLoss on training set
- **Validation Loss**: CrossEntropyLoss on validation set
- **Test Loss**: Final evaluation on held-out test set
- **Accuracy**: Not currently computed (commented out in training loop)

### Expected Performance

Based on git history and code comments:
- Models trained on 100k-500k positions
- Checkpoints saved at regular intervals
- Model can play basic chess but quality depends on training data

## Dependencies

**Core Libraries**:
- `torch`: PyTorch for neural networks
- `torch_geometric`: Graph neural network library
- `python-chess`: Chess logic and FEN parsing
- `hydra-core`: Configuration management
- `mlflow`: Experiment tracking (optional)
- `pandas`: Data processing
- `numpy`: Numerical operations
- `tqdm`: Progress bars
- `scikit-learn`: Class weight computation

**Installation**:
```bash
pip install torch torch_geometric python-chess hydra-core mlflow pandas numpy tqdm scikit-learn
```

## MLflow Integration

**Configuration**:
- Set `TRACKING_URL` environment variable for remote tracking
- Currently disabled by default (`tracking_url = False` in `graph.py:355`)

**Logged Metrics**:
- Hyperparameters: epochs, batch_size, learning_rate
- Training loss per epoch
- Validation loss per epoch
- Model artifacts

## Known Issues and TODOs

1. **Preprocessing parallelization** (`dataset.py:71`): Not implemented
2. **Hardcoded CUDA** (`model.py:126-127`): Should use device parameter
3. **Accuracy calculation disabled** (`graph.py:409-420`): Commented out
4. **Checkpointing path** (`graph.py:433`): Hardcoded directory structure
5. **MLflow disabled**: `tracking_url = False` overrides environment variable
6. **Manual material calculations**: Could use python-chess built-ins

## Chess-Specific Concepts

### FEN Notation

Format: `<pieces> <turn> <castling> <en_passant> <halfmove> <fullmove>`

Example: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`
- Pieces: lowercase=black, uppercase=white, numbers=empty squares
- Turn: `w` or `b`
- Castling: `KQkq` (white kingside, white queenside, black kingside, black queenside)
- En passant: target square or `-`

### Piece Values (Centipawns)

Standard material values used in `win_rate_params()`:
- Pawn: 100 (1.0)
- Knight: 300 (3.0)
- Bishop: 300 (3.0)
- Rook: 500 (5.0)
- Queen: 900 (9.0)
- King: Infinite (not counted)

### Win Rate Model

Based on Stockfish's UCI implementation, considers:
1. **Evaluation score**: Centipawn advantage
2. **Material count**: Positions with more material are more drawish
3. **Logistic curve**: Maps evaluation to probability in [0, 1]

Formula: `P(win) = 1 / (1 + exp((a - v) / b))`
- `a`, `b`: Material-dependent parameters
- `v`: Normalized evaluation score

## Quick Reference

### File Purposes at a Glance

| File | Primary Function | When to Edit |
|------|------------------|--------------|
| `config/default.yaml` | Hyperparameters, model selection | Changing model, features, or data |
| `fen_parser.py` | Parse FEN strings | Adding new chess-specific features |
| `graph.py` | Training and inference | Main training loop, graph construction |
| `model.py` | GNN architectures | Adding new model architectures |
| `dataset.py` | Data loading | Changing schema or loading logic |
| `preprocess.py` | Data preprocessing | Changing binning strategy |
| `util.py` | Helper functions | Adding normalization or utility functions |
| `graph.ipynb` | Experimentation | Interactive exploration and debugging |

### Important Constants

- **Board squares**: 64 (8x8)
- **Node features**: 15
- **Edge features**: 12
- **Default bins**: 16 (8 for white advantage, 8 for black advantage)
- **Training split**: 90% train, 5% val, 5% test
- **Default batch size**: 32
- **Default learning rate**: 0.001
- **Training epochs**: 200
- **Checkpoint frequency**: Every 10 epochs
- **DataLoader workers**: 6

### Key Functions

- `create_edges()`: Static graph structure creation
- `fen_to_node_features()`: Convert position to node features
- `fen_to_edge_features()`: Convert legal moves to edge features
- `create_graph_from_fen()`: Build complete graph object
- `parse_evaluation()`: Convert evaluation strings to centipawns
- `win_rate_model()`: Evaluation to win probability
- `win_rate_to_bin()`: Probability to discrete class
- `load_datasets()`: Create train/val/test dataloaders
- `train()`: Main training function (Hydra entry point)
- `infer()`: Single position evaluation

## Tips for AI Assistants

1. **Always read before editing**: This codebase has many interconnected pieces. Check how features dimensions are used across files.

2. **Test with small data**: When adding features, test with a small dataset first to verify dimensions match.

3. **Check device placement**: Mix of CPU/CUDA code. Be consistent when adding new tensors.

4. **Update all dimension references**: Changing feature dimensions requires updates in:
   - `config/default.yaml`
   - `dataset.py` schema
   - `model.py` encoders
   - `fen_parser.py` or `graph.py` feature functions

5. **Consider chess semantics**: Features should make sense from chess perspective (e.g., attacked squares, piece mobility, king safety).

6. **Preserve FEN compatibility**: Any changes to parsing should maintain standard FEN format.

7. **Document evaluation conventions**: Be clear about sign conventions (positive=white advantage) when adding features.

8. **Test both classification and regression**: Code supports both, so changes should work with either loss function.

9. **Mind the batch dimension**: PyTorch Geometric uses special batching. The `batch` tensor indicates which graph each node belongs to.

10. **Use git history**: Recent commits show evolution of feature engineering and architecture decisions.
