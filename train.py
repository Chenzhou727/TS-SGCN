# train.py
import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from SeisDataloader import make_dataloaders
from loss import *
from model import MagRegressor
from utils import EarlyStopping, save_checkpoint, plot_history, load_config
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, epoch, device):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}" if epoch is not None else "Train", disable=True)

    for batch_idx, batch in enumerate(pbar):

        waveforms, mags = batch
        waveforms = waveforms.to(device)
        mags = mags.to(device)
    
        optimizer.zero_grad()
    
        out = model(waveforms)
    
        loss = weightedHuberLoss(out, mags)
    
        loss.backward()
        optimizer.step()
    
        total_loss += float(loss.item()) * waveforms.size(0)
        n += waveforms.size(0)
    return total_loss / n


def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            waveforms, mags = batch
            waveforms = waveforms.to(device)
            mags = mags.to(device)
            out = model(waveforms)
            loss = weightedHuberLoss(out, mags)
            total_loss += float(loss.item()) * waveforms.size(0)
            n += waveforms.size(0)
    return total_loss / n


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load data
    train_loader, val_loader = make_dataloaders(dataset=args.dataset, batch_size=args.batch_size,
                                               num_workers=args.num_workers, seed=args.seed)

    # model
    model = MagRegressor(window=args.window, stride=args.stride, n_fft=args.n_fft, hop_length=args.hop_length, fc=args.fc,
                         filters=args.filters, embedding_dims=args.embedding_dims, num_heads=args.num_heads, 
                         dropout=args.dropout)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=4, min_lr=5e-7)
    early_stopper = EarlyStopping(patience=args.early_stopping)

    save_dir = os.path.join(os.getcwd(), args.model_path)
    os.makedirs(save_dir, exist_ok=True)

    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):

        start_time = time.time()  # Record start time
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, device)
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Time per training epoch: {elapsed_time:.2f} s")
        val_loss = eval_one_epoch(model, val_loader, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss
            }, is_best, checkpoint_dir=save_dir, filename=f'ckpt_epoch_{epoch}.pt')

        if early_stopper.step(val_loss):
            print("Early stopping triggered.")
            break

    plot_history(history, save_dir)
    print("Training finished. Best val loss:", best_val)


if __name__ == "__main__":
    cfg = load_config()
    main(cfg)

