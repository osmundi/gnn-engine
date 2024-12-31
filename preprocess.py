import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import hydra
from omegaconf import DictConfig

from fen_parser import FenParser
from util import win_rate_model, win_rate_to_bin


def write_with_classes(d, base_dir, file_name):
    df = pd.DataFrame(data=d)
    write_data = f'{base_dir}/{file_name}.csv'
    df.to_csv(write_data, index=False)


def create_split_dict(dataset, base_path, file_name):
    total_rows = len(dataset)
    train_ratio = 0.7
    val_ratio = 0.15

    # Compute the lengths
    train_len = int(total_rows * train_ratio)
    val_len = int(total_rows * val_ratio)

    ranges = {
        'train': torch.arange(0, train_len),
        'val': torch.arange(train_len, train_len + val_len),
        'test': torch.arange(train_len + val_len, total_rows)
    }

    # Save the dictionary to a file
    write_path = f'{base_path}/ranges/{file_name}.pt'
    torch.save(ranges, write_path)


def class_weights(data, base_path, file_name):
    classes = np.unique(data)
    class_weights = compute_class_weight(
        'balanced', classes=classes, y=data)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    write_path = f'{base_path}/class_weights/{file_name}.pt'
    torch.save(class_weights, write_path)


def plot(data):
    # Group by unique values and count occurrences

    df = pd.DataFrame(data=data)
    grouped = df.value_counts().sort_index()

    # Plot the grouped data
    grouped.plot(kind="bar", title="Grouped Values Plot",
                 xlabel="Value", ylabel="Count", color="skyblue")
    plt.grid(axis='y')
    plt.show()



@hydra.main(version_base=None, config_path="config", config_name="default")
def preprocess(cfg: DictConfig):
    base_dir = 'dataset/raw'
    file_name = 'first_100k_evaluations'
    raw_data = f'{base_dir}/unprocessed/{file_name}.csv'

    bins = cfg.data.bins / 2

    # process raw data
    df = pd.read_csv(raw_data)
    iterator = enumerate(zip(df['fen'], df['evaluation']))
    d = {'fen': [], 'evaluation': []}
    for _, (fen, evaluation) in tqdm(iterator, total=len(df)):
        fen_parser = FenParser(fen)

        d['fen'].append(fen)

        if evaluation.startswith('#'):
            # win rate probability 100%
            bin = bins - 1
        else:
            bin = win_rate_to_bin(win_rate_model(
                abs(int(evaluation)), fen_parser.piece_counts()), bins)

        # use negative evaluations in classes
        # if black to move and original evaluation is positive
        # OR if white to move and original evaluation is negative
        # -> multiply bin by -1
        if fen_parser.white_to_move() and (evaluation.startswith('#-') or evaluation.startswith('-')):
            bin += bins
            #bin = bin * -1
        if not fen_parser.white_to_move() and (evaluation.startswith('#+') or evaluation.startswith('+')):
            bin += bins

        d['evaluation'].append(bin)

    # write processed data to a new csv dataset
    write_with_classes(d, base_dir, file_name)

    # balance classes with weights
    class_weights(d['evaluation'], base_dir, file_name)

    # create indices for train/val/test
    create_split_dict(df, base_dir, file_name)

    # plot classes
    plot(d['evaluation'])


if __name__ == '__main__':
    preprocess()
