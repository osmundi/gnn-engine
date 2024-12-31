
import torch
import numpy as np
import pandas as pd


def normalize_evalutions(chess_data):

    data = pd.read_csv(chess_data, header=None, names=["fen", "evaluation"])

    # Assuming "evaluation" is the second column after splitting
    data['evaluation'] = data['evaluation'].apply(
        lambda x: 1000 if '#+' in x else (-1000 if '#-' in x else int(x.strip()))
    )

    # Convert evaluations to a PyTorch tensor
    return torch.tensor(data['evaluation'].values, dtype=torch.float)


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


def normalize_evaluation(evaluation: int, evaluations) -> float:

    # Compute median and IQR
    median = torch.median(evaluations)
    q1 = torch.quantile(evaluations, 0.25)
    q3 = torch.quantile(evaluations, 0.75)
    iqr = q3 - q1

    # Robust scaling
    robust_scaled_evaluations = (evaluation - median) / iqr

    # Optionally, scale robust outputs to [-1, 1] by dividing further
    return torch.tanh(robust_scaled_evaluations)  # Limits to [-1, 1]


def denormalize_evaluation(normalized_data: float, evaluations) -> float:
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
        position.get('P', 0)
        + 3 * position.get('N', 0)
        + 3 * position.get('B', 0)
        + 5 * position.get('R', 0)
        + 9 * position.get('Q', 0)
    )

    # Clamp material count and normalize
    m = np.clip(material, 17, 78) / 58.0

    # Coefficients for the polynomial model
    as_c = [-37.45051876, 121.19101539, -132.78783573, 420.70576692]
    bs_c = [90.26261072, -137.26549898, 71.10130540, 51.35259597]

    a = as_c[0] * m**3 + as_c[1] * m**2 + as_c[2] * m + as_c[3]
    b = bs_c[0] * m**3 + bs_c[1] * m**2 + bs_c[2] * m + bs_c[3]

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
    """Binning: divide winrates between [0,1] to a N classes"""
    return min(max(0, int(rate * (bins - 1))), bins - 1)
