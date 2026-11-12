# utils.py
import os
import torch
import matplotlib.pyplot as plt
import argparse
import yaml


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, metric):
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
            return False  # do not stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pt'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Do not save the model at every step; only save the best one
    # path = os.path.join(checkpoint_dir, filename)
    # torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(state, best_path)


def plot_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss', linestyle='--')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'))
    plt.close()


def load_config():
    parser = argparse.ArgumentParser()

    # Config file
    parser.add_argument("--config", type=str, default="config.yaml")

    # Overridable parameters
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)

    parser.add_argument("--window", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--n_fft", type=int)
    parser.add_argument("--hop_length", type=int)

    parser.add_argument("--fc", type=int)
    parser.add_argument("--dropout", type=float)

    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--embedding_dims", type=int)

    parser.add_argument("--model_path", type=str)

    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--early_stopping", type=int)


    parser.add_argument("--regressor", type=str)
    args = parser.parse_args()

    # Read the config file
    config_path = args.config
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Command-line overrides
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
    config = argparse.Namespace(**config)
    return config


