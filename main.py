import gc
import os

import config
import utils
from dataset import DKTDataset
from lana_arch import LANA

import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.metrics import roc_auc_score

import argparse

# Config
tqdm.pandas()
utils.set_seed(config.SEED)


# Pytorch Lightning Module
class TorchModel(pl.LightningModule):
    def __init__(self, trainer_args, model_args):
        super().__init__()
        self.model = LANA(**model_args)
        self.val_labels = []
        self.val_outs = []

    def forward(self, input):
        return self.model(input)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.LEARNING_RATE)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        target_mask = (inputs["content_id"] != config.PAD)
        output = self(inputs)
        output = torch.masked_select(output, target_mask)
        target = torch.masked_select(target, target_mask)

        loss = nn.BCEWithLogitsLoss()(output.float(), target.float())
        auc = roc_auc_score(target.cpu(), output.detach().float().cpu())
        self.log("t_loss", loss)
        self.log("t_auc", auc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        target_mask = (inputs["content_id"] != config.PAD)
        output = self(inputs)
        output = torch.masked_select(output, target_mask)  # probability
        target = torch.masked_select(target, target_mask)
        loss = nn.BCEWithLogitsLoss()(output.float(), target.float())
        auc = roc_auc_score(target.cpu(), output.detach().float().cpu())
        self.val_labels.extend(target.view(-1).data.cpu().numpy())
        self.val_outs.extend(output.view(-1).data.cpu().numpy())
        self.log("v_loss", loss, prog_bar=True)
        self.log("v_auc", auc, prog_bar=True)

    def on_validation_epoch_end(self):
        real_auc = roc_auc_score(self.val_labels, self.val_outs)
        self.log("v_auc", real_auc, prog_bar=True)
        self.val_labels = []
        self.val_outs = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LANA")
    parser.add_argument('-d', '--data', type=str, required=True,
                        help="Filepath of the preprocessed data")

    args = parser.parse_args()
    train_df = pd.read_pickle(f"{args.data}.train")
    val_df = pd.read_pickle(f"{args.data}.valid")
    print("train size: ", train_df.shape, "validation size: ", val_df.shape)

    train_dataset = DKTDataset(train_df.values, max_seq=config.MAX_SEQ,
                               min_seq=config.MIN_SEQ, overlap_seq=config.OVERLAP_SEQ)
    val_dataset = DKTDataset(val_df.values, max_seq=config.MAX_SEQ,
                             min_seq=config.MIN_SEQ, overlap_seq=config.OVERLAP_SEQ)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              num_workers=8,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            num_workers=8,
                            shuffle=False,
                            pin_memory=True)
    del train_dataset, val_dataset
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

    if not os.path.exists("./saved_models"):
        os.mkdir("./saved_models")
    checkpoint = ModelCheckpoint(dirpath="./saved_models",
                                 filename="model-{epoch}-{v_auc:.2f}",
                                 verbose=True,
                                 save_top_k=1,
                                 save_last=True,
                                 mode="max",
                                 monitor="v_auc")

    lana_model = TorchModel(trainer_args=args, model_args=ARGS)
    if config.DEVICE is None or not torch.cuda.is_available():
        trainer = pl.Trainer(progress_bar_refresh_rate=1,
                             max_epochs=config.EPOCH, callbacks=[checkpoint])
    else:
        trainer = pl.Trainer(progress_bar_refresh_rate=1,
                             max_epochs=config.EPOCH, callbacks=[checkpoint],
                             gpus=config.DEVICE)
    trainer.fit(model=lana_model,
                train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint("./saved_models/final.pt")