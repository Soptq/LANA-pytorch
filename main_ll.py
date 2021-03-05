import gc
import copy
import os
import math

import config
import utils
import argparse
from dataset_group import DKTDataset
from lana_arch import LANA

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import trange

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Config
tqdm.pandas()
utils.set_seed(config.SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leveled Learning")
    parser.add_argument('-d', '--data', type=str, required=True,
                        help="Filepath of the preprocessed data")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Saved model parameters by Pytorch-Lightning")
    parser.add_argument('-n', '--num', type=int, required=True,
                        help="Number of levels")
    parser.add_argument('-t', '--top', type=int, required=True,
                        help="Top-k")
    parser.add_argument('-i', '--mu_itv', type=float, required=True,
                        help="Mean interval")

    args = parser.parse_args()

    n_models = args.num
    train_df = pd.read_pickle(f"{args.data}.train")
    val_df = pd.read_pickle(f"{args.data}.valid")
    user_performance = utils.read_pickle(f"{args.data}.user")

    print("train size: ", train_df.shape, "validation size: ", val_df.shape)

    train_loaders = []
    print("Generating dataset...")
    train_dataset = DKTDataset(train_df, max_seq=config.MAX_SEQ,
                               min_seq=config.MIN_SEQ, overlap_seq=config.OVERLAP_SEQ,
                               user_performance=user_performance, n_levels=n_models, mu_itv=args.mu_itv)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              num_workers=8,
                              shuffle=False,
                              pin_memory=True)
    val_dataset = DKTDataset(val_df, max_seq=config.MAX_SEQ,
                             min_seq=config.MIN_SEQ, overlap_seq=config.OVERLAP_SEQ,
                             user_performance=user_performance, n_levels=n_models, mu_itv=args.mu_itv)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            num_workers=8,
                            shuffle=False,
                            pin_memory=True)
    print(f"All dataloaders are generated")
    del train_dataset
    del val_dataset
    gc.collect()

    ARGS = {"d_model": config.MODEL_DIMS,
            'n_head': config.N_HEADS,
            'n_encoder': config.NUM_ENCODER,
            'n_decoder': config.NUM_DECODER,
            'dim_feedforward': config.FEEDFORWARD_DIMS,
            'dropout': config.DROPOUT,
            'max_seq': config.MAX_SEQ,
            'n_exercises': config.TOTAL_EID,
            'n_parts': config.TOTAL_PART,
            'n_resp': config.TOTAL_RESP,
            'n_etime': config.TOTAL_ETIME,
            'n_ltime_s': config.TOTAL_LTIME_S,
            'n_ltime_m': config.TOTAL_LTIME_M,
            'n_ltime_d': config.TOTAL_LTIME_D}

    DEVICE = f"cuda:{config.DEVICE[0]}" if torch.cuda.is_available() else "cpu"
    models = [LANA(**ARGS).to(DEVICE) for _ in range(n_models)]
    optimizers = [torch.optim.AdamW(models[i].parameters(), lr=config.LEARNING_RATE / 1000) for i in range(n_models)] # Finetune

    baseline, optimizer_state, model_state = utils.load_model(args.model, device=DEVICE)
    model_state = utils.remove_prefix_from_dict(model_state, "model.")  # remove prefix added by pytorch-lightning

    print("Baseline AUC: ", baseline)
    print("Loading models...")
    for cluster in range(n_models):
        models[cluster].load_state_dict(model_state)

    print("All models loaded")
    for epoch in range(config.EPOCH):
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            t.set_description(f"Epoch {epoch}")
            for batch_idx, batch in enumerate(train_loader):
                for cluster in range(n_models):
                    models[cluster].train()
                    optimizers[cluster].zero_grad()
                inputs, target, probs = batch
                inputs = utils.dict_to_device(inputs, DEVICE)
                target = target.to(DEVICE)
                target_mask = (inputs["content_id"] != config.PAD)
                probs = probs.to(DEVICE)
                total_loss = 0.
                target = torch.masked_select(target, target_mask)
                for cluster in range(n_models):
                    output = models[cluster](inputs)
                    output = torch.masked_select(output, target_mask)
                    weight = torch.masked_select(probs[:, cluster].unsqueeze(-1), target_mask)
                    loss = nn.BCEWithLogitsLoss(reduction="none")(output.float(), target.float())
                    loss = loss * weight
                    loss = loss.mean()
                    total_loss += loss.item()
                    loss.backward()
                    optimizers[cluster].step()
                t.set_postfix({"t_loss": ("%.4f" % total_loss)})
                t.update()
        for cluster in range(n_models):
            models[cluster].eval()

        with torch.no_grad():
            val_labels = []
            val_outs = []
            with tqdm(total=len(val_loader), dynamic_ncols=True) as t:
                t.set_description("Validating")
                for batch_idx, batch in enumerate(val_loader):
                    inputs, target, probs = batch
                    inputs = utils.dict_to_device(inputs, DEVICE)
                    target = target.to(DEVICE)
                    target_mask = (inputs["content_id"] != config.PAD)
                    probs = probs.to(DEVICE)
                    probs = utils.prob_topk(probs, args.top)
                    output = torch.zeros(target.shape, device=DEVICE)
                    for cluster in range(n_models + 1):
                        m_output = models[cluster](inputs)
                        output += (m_output * probs[:, cluster].unsqueeze(-1))

                    output = torch.masked_select(output, target_mask)
                    target = torch.masked_select(target, target_mask)
                    loss = nn.BCEWithLogitsLoss()(output.float(), target.float())
                    auc = roc_auc_score(target.cpu(), output.detach().float().cpu())
                    val_labels.extend(target.view(-1).data.cpu().numpy())
                    val_outs.extend(output.view(-1).data.cpu().numpy())
                    t.set_postfix({"v_loss": ("%.4f" % loss.item()), "v_auc": ("%.4f" % auc)})
                    t.update()
            real_auc = roc_auc_score(val_labels, val_outs)
            print(f"AUC: ", real_auc)
