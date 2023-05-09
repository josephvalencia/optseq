import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl

# define the LightningModule
class APARENT(pl.LightningModule):
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=(4,8),padding='valid')
        self.cnn2 = nn.Conv2d(in_channels=96,out_channels=128,kernel_size=(1,6),padding='valid')
        self.pool = nn.MaxPool2d(kernel_size=8,dilation=2) 
        self.fc = nn.Linear(256,1)
        self.dropout = nn.Dropout(p=0.2) 
        self.stack = nn.Sequential(self.cnn1,
                                    nn.ReLU(),
                                    self.pool,
                                    self.cnn2,
                                    nn.ReLU())
        self.output = nn.Sequential(self.fc,
                                    nn.ReLU(), 
                                    self.dropout)
    
    def symmetric_kl_divergence_loss(self, p_pred, p_obs):
        a =  p_obs * torch.log(p_obs / p_pred) 
        b =  (1 - p_obs) * torch.log((1 - p_obs) / (1 - p_pred)) 
        kl = a + b 
        return kl.mean()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, p_obs = batch
        z = self.stack(x).reshape(-1)
        p_pred = nn.Sigmoid(self.output(z))
        loss = self.symmetric_kl_divergence_loss(p_pred, p_obs) 
        self.log("train_loss", loss)
        return loss
    
    def valid_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, p_obs = batch
        z = self.stack(x).reshape(-1)
        p_pred = nn.Sigmoid(self.output(z))
        loss = self.symmetric_kl_divergence_loss(p_pred, p_obs) 
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == '__main__':

    test_df,train_df = parse_egfp_polysome("data/5-prime_sample_etal/GSM3130435_egfp_unmod_1.csv","data/5-prime_sample_etal/GSM3130436_egfp_unmod_2.csv")
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
    optimus = OptimusFivePrime()
    module = MeanRibosomeLoadModule(optimus) 
    wandb_logger = pl.loggers.WandbLogger(project="optimus_5prime")
    trainer = pl.Trainer(max_epochs=3,devices=1,
                         accelerator="gpu",logger=wandb_logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(module,train_loader,val_loader)
    trainer.test(module,test_loader)
    
    # init the autoencoder
    aparent = APARENT()
