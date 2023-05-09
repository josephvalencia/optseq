from torch import optim, nn
import lightning.pytorch as pl
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from models.optimus_5prime.polysome import *

if __name__ == "__main__":

    test_df,train_df = parse_egfp_polysome("data/5-prime_sample_etal/GSM3130435_egfp_unmod_1.csv","data/5-prime_sample_etal/GSM3130436_egfp_unmod_2.csv")
    print(add_mrl(train_df)[['rl','mrl']].describe()) 
    quit() 
    test_set,valid_set,train_set = make_dataset_splits(test_df,train_df,42)
    train_loader = dataloader_from_dataset(train_set,128)
    val_loader = dataloader_from_dataset(valid_set,128) 
    test_loader = dataloader_from_dataset(test_set,128)

    checkpoint_callback = ModelCheckpoint(
                        dirpath="models/optimus_5prime/checkpoints",
                        monitor="val_loss",
                        save_top_k=3,
                        every_n_epochs=1,
                        filename="{epoch}-{val_loss:.4f}")

    early_stopping = EarlyStopping("val_loss",patience=5)

    # train the model
    module = MeanRibosomeLoadModule() 
    wandb_logger = pl.loggers.WandbLogger(project="optimus_5prime")
    trainer = pl.Trainer(max_epochs=3,devices=1,
                         accelerator="gpu",logger=wandb_logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(module,train_loader,val_loader)
    trainer.test(module,test_loader)
