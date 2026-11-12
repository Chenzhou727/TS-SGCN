# predict.py
import argparse
import torch
import numpy as np
from model import MagRegressor
from SeisDataloader import make_dataset
from utils import load_config
from datetime import datetime

def load_model(args, device):
    # build model
    model = MagRegressor(window=args.window, stride=args.stride, n_fft=args.n_fft, hop_length=args.hop_length, fc=args.fc,
                         filters=args.filters, embedding_dims=args.embedding_dims, num_heads=args.num_heads, dropout=args.dropout)
    model.to(device)
    ckpt = torch.load(args.model_path+'/'+args.test_model, map_location=device, weights_only=False)
    print("Current model epoch:", ckpt["epoch"])
    # ckpt might be state dict or dict with 'state_dict'
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    return model


def dataset_to_tensor(dataset):
    # load all data into a single tensor (be careful with memory)
    waveforms = dataset.waveforms
    mags = dataset.mags
    metas = dataset.metas

    return waveforms, mags, metas


def predict(model, x_tensor, y_tensor, metas, device, n_iter=50, batch_size=256):
    """
    x_tensor: torch.Tensor (N, C, L)
    return: y_mean, aleatoric_mean, epistemic_std, combined_var
    """
    model.eval()
    preds = []
    dataset = torch.utils.data.TensorDataset(x_tensor,y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # with torch.no_grad():
    #     batch_MAE = []
    #     for (waveforms, mags) in loader:
    #         waveforms = waveforms.to(device)
    #         mags =  mags.to(device)
    #         out = model(waveforms)  # (B,1)
    #         MAE = torch.abs(mags - out)
    #         batch_MAE.append(MAE)
    #     batch_MAE = torch.vstack(batch_MAE)  # (N,`1`)
    # y_mean = batch_MAE.mean()
    with torch.no_grad():
        batch_MAE = []
        all_mags = []
        all_outs = []
        for waveforms, mags in loader:
            waveforms = waveforms.to(device)
            mags = mags.to(device)
            out = model(waveforms)      # (B,1)
            MAE = torch.abs(mags - out) # (B,1)
            batch_MAE.append(MAE)
            all_mags.append(mags)
            all_outs.append(out)
    
        batch_MAE = torch.vstack(batch_MAE)  # (N,1)
        all_mags = torch.vstack(all_mags)    # (N,1)
        all_outs = torch.vstack(all_outs)    # (N,1)
        

        # Option 1: Convert to numpy for display/export
        import pandas as pd
        df = pd.DataFrame({
            "meta": metas,
            "mag": all_mags.squeeze(1).cpu().numpy(),
            "pred": all_outs.squeeze(1).cpu().numpy()
        })
    
    MAE = batch_MAE.mean()
    RMSE = torch.sqrt(torch.mean(batch_MAE ** 2))
    return MAE,RMSE,df



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # we will use test split from prepare_splits via data.make_dataloaders, but to be simple:

    print(f"Loading test set: ")
    dataset = make_dataset(args.dataset, "test")
    waveforms, mags, metas= dataset_to_tensor(dataset)
    waveforms = waveforms.to(device)
    mags = mags.to(device)

    model = load_model(args, device)
    # predict
    pred_MAE,pred_RMSE,df = predict(model, waveforms, mags, metas, device, n_iter=args.n_mc, batch_size=args.batch_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.dataset}_{args.model_path}_{timestamp}.csv"
    df.to_csv(filename, index=False)

    print("Predictions MAE:", pred_MAE)
    print("Predictions RMSE:", pred_RMSE)


if __name__ == "__main__":
    cfg = load_config()
    main(cfg)
