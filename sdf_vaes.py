from third_party.sdf_vae.models import *
from abc import abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from functools import partial, wraps
# from datasets import *
from tqdm.auto import tqdm
import trimesh
import skimage
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
import wandb
from utils import Query
import argparse


class VAEPointCloudSDFModel(pl.LightningModule):
    """
    Encode point clouds using a transformer, and query points using cross
    attention to the encoded latents.
    """

    def __init__(
        self,
        args
    ):
        super().__init__()
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
            
        self.args=args

        self.specs = {"SdfModelSpecs" : {
                "hidden_dim" : 512,
                "latent_dim" : 32,
                "pn_hidden_dim" : 128,
                "num_layers" : 12,
                "vae":args.with_ae
                },
                    
                "SampPerMesh" : 48000,
                "PCsize" : 1024,
            
                "kld_weight" : 1e-4 if int(args.vae) else 1e-4,
                "latent_std" : 1. if int(args.vae) else "zero_mean" ,
            
                "sdf_lr" : 1e-4,
                "training_task":"modulation",
                }
        
        self.model =  CombinedModel(self.specs)
        
        self.save_hyperparameters(self.specs)
        self.save_hyperparameters(args)
        self.train_testing = None
        

    def encode_point_clouds(self, point_clouds: torch.Tensor, return_distribution=False):
        plane_features = self.model.sdf_model.pointnet.get_plane_features(point_clouds)
        mu, logvar = self.model.vae_model.encode(torch.cat(plane_features, dim=1))
        if return_distribution:
            return [mu, logvar]
        return self.model.vae_model.reparameterize(mu, logvar)
    
    def encode_to_triplane(self, point_clouds: torch.Tensor):
        plane_features = self.model.sdf_model.pointnet.get_plane_features(point_clouds)
        return torch.cat(plane_features, 1)
        
    
    def get_loss(self, query, point_clouds, sdf, epoch, reduction="none"):
        x = {"xyz":query, "gt_sdf":sdf, "point_cloud":point_clouds, "epoch":epoch, "reduction":reduction}
        return self.model.train_modulation(x)
        
    
    def predict_occupancy(
        self, x: torch.Tensor, encoded:torch.Tensor
    ) -> torch.Tensor:
        encoded =  self.model.vae_model.decode(encoded) #if len(encoded.shape) <=2 else self.model.vae_model(encoded)[0]
        pred_sdf = self.model.sdf_model.forward_with_plane_features(encoded, x)
        return pred_sdf
    
    def forward(
        self,
        x: torch.Tensor,
        point_clouds: torch.Tensor,
        sdf:torch.Tensor,
        epoch=100
    ) -> torch.Tensor:
        return self.get_loss(x, point_clouds, sdf, epoch=epoch)
    
    
    def training_step(self, batch, batch_idx):
        data = batch
        device = self.device
      
        loss, _ = self(data["points"].to(device), data["pc"].to(device), data["sdf"].to(device), epoch=1000)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.99))
        return optimizer


    def train_dataloader(self):
        train_ds = CollectDataset(path_to_data=self.args.path_to_data, number_of_data=args.training_size, device="cpu")
        data= DataLoader(train_ds, batch_size=self.args.train_batch_size, shuffle=True, num_workers=16)
        self.train_testing = next(iter(data))
        return data

    def val_dataloader(self):
        val_ds = CollectDataset(path_to_data=self.args.path_to_data, number_of_data=args.training_size, device="cpu", partition="test",)
        self.temp = Query(grid_size=64, device="cpu")
        return DataLoader(val_ds, batch_size=self.args.val_batch_size, pin_memory=True, shuffle=True, num_workers=8, drop_last=True)
    
    def validation_step(self, batch, batch_idx):
        data = batch
        device = self.device

        with torch.no_grad():
            loss, losses = self(data["points"].to(device), data["pc"].to(device), data["sdf"].to(device), epoch=self.current_epoch)

        # You might want to compute some validation metrics here and return them
        # For example:
        # val_loss = some_loss_function(predicted_images, image)
        # return {"val_loss": val_loss, "predicted_images": predicted_images, "true_images": image}
        # Temporarily returning an empty dict until you define your metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {}

    def on_validation_epoch_end(self, outputs=None):
   
        self.model.eval()
        with torch.no_grad():
            data = next(iter(self.val_dataloader()))  # Get a batch from the validation set
            self.sample(data, "test_Sample")
            
        if self.train_testing is not None:
            with torch.no_grad():
                self.sample(self.train_testing, "train_Samples")
                
        self.model.train()
 
    def sample(self, data, name="test"):
        pc_to_log = []
        pc=data["pc"]
        device = self.device
        with torch.no_grad():
            predicted_sdf_latent = self.encode_point_clouds(data["pc"].to(self.device))
        
        sdfs = []
                   
        for i in range(0, len(self.temp), 48000):
            cur = self.temp[i:i+48000][None].expand(len(predicted_sdf_latent), -1, -1).to(self.device)
            sdf = self.predict_occupancy(cur, predicted_sdf_latent)
            sdfs.append(sdf.cpu())
            
        sdfs = torch.cat(sdfs, 1)
        
        data=[]
        columns = ["True Object", "Predicted Object"]
        for i, sdf in enumerate(sdfs):
            
            sdf = sdf-sdf.mean() if torch.all(sdf < 0) or torch.all(sdf > 0) else sdf
            sdf = sdf.view(self.temp.grid_size, self.temp.grid_size, self.temp.grid_size).cpu().numpy()
            
            verts, faces, normals, _ = skimage.measure.marching_cubes(
            volume=sdf,
            level=0,
            allow_degenerate=False,
            spacing=(self.temp.voxel_size,) * 3,
            )
            
            # The triangles follow the left-hand rule, but we want to
            # follow the right-hand rule.
            # This syntax might seem roundabout, but we get incorrect
            # results if we do: x[:,0], x[:,1] = x[:,1], x[:,0]
            old_f1 = faces[:, 0].copy()
            faces[:, 0] = faces[:, 1]
            faces[:, 1] = old_f1
            
            verts += self.temp.min_coord
            
            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                normals=normals,
            )

            points, _ = trimesh.sample.sample_surface(mesh, 4096)
            
            d = [wandb.Object3D(pc[i].numpy()),  wandb.Object3D(points)]
            data.append(d)
            
        self.logger.log_table(key=name, columns=columns, data=data)
        
        
        # wandb.log({f"True point_cloud{i+1}": wandb.Object3D(pc[i].numpy()), f"Recons. pc{i+1}": wandb.Object3D(points), }, step=self.current_epoch)             
        # wandb.log({f"True Image {i+1}": wandb.Image(image[i]), f"pred image{i+1}":wandb.Image(predicted_images[i])}, step=self.current_epoch )
    
    
def main(args):
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    # Initialize WandB logging
    name=args.experiment_name
    wandb_logger = WandbLogger(project="OccluderGen", name=name)
    
    model = VAEPointCloudSDFModel(args)
    
    # model = model.load_from_checkpoint("/data/fraji/checkpoints/SSD/model-epoch=44-val_loss=0.19.ckpt")
    model.args.train_batch_size=args.train_batch_size
    

    # Setup your ModelCheckpoint callback as previously described
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.results_folder,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        save_top_k=3,
        mode='min',
        
    )

    # Configure the Trainer to use 3 GPUs
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu',  # Specify 'gpu' as the accelerator
        devices=3,  # Use 3 GPUs
        strategy='ddp',  # Use Distributed Data Parallel for multi-GPU training
        check_val_every_n_epoch=20,
    )
    # Train the model
    trainer.fit(model)

if __name__ == "__main__":
    
    import random
    import argparse
    
    # Datasets and loaders
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--path_to_data', type=str, default="/data/fraji/sdfs",)                   
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--training_size', type=str, default=-1)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--val_freq', type=int, default=2000)
    parser.add_argument('--vae', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--with_ae', type=int, default=0)
    parser.add_argument('--results_folder', type=str, default="/data/fraji/Estimate_3D/implicit/checkpoints/sdf_vae")
    parser.add_argument('--experiment_name', type=str, default="Implicit_3D")
    parser.add_argument('--seed', type=int, default=2024)
    
    def seed_all(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    args = parser.parse_args()
    seed_all(args.seed)
    
    main(args)
    
    