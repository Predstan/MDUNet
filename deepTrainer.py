


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datasets_occluder import CoreDataset
from torch.utils.data import DataLoader
from model import NlosModel
from utils import AverageMeter, awgn, sbr
import trimesh
import skimage
import numpy as np
import argparse
import torch
from torch.optim import Adam, AdamW
import wandb


def cycle(dl):
    while True:
        for data in dl:
            yield data

class NlosModelLightning(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
            
        self.args = args
        self.model = NlosModel(sdf_path=args.sdf_path, unet=args.unet, fine_tune_image=args.fine_tune_image,
        fine_tune_sdf=args.fine_tune_sdf).to(self.device)
        self.train_testing = None
        
    def forward(self, data):
        device = self.device
        r = np.random.choice([0, 1], p=[0.5, 0.5])
        condition_seq = awgn(data["measurements"].permute(0, 3, 1, 2).to(device), 15, 100) if r else sbr(data["measurements"].permute(0, 3, 1, 2).to(device), 15, 100)
        training_seq = data["pointclouds"].to(device)
        image = data["image"].to(device)
        # Forward pass through your model
        return self.model(condition_seq, image, training_seq)

    def train_dataloader(self):
        train_ds = CoreDataset(
            path_to_data=self.args.path_to_data, 
            root_to_image="/data/fraji/ImageNet-Datasets-Downloader/imagenet",
            image_size=256, seq_length=2048, augument=True,
            image_channel=3, partition="train",)
        data = DataLoader(train_ds, batch_size=self.args.train_batch_size, shuffle=True, num_workers=12)
        self.train_testing = next(iter(data))
        return train_ds, data
    
    def val_dataloader(self):
        val_ds = CoreDataset(
            path_to_data=self.args.path_to_data, 
            root_to_image="/data/fraji/ImageNet-Datasets-Downloader/imagenet",
            image_size=256, seq_length=2048, augument=True,
            image_channel=3, partition="test",)
        return val_ds, DataLoader(val_ds, batch_size=self.args.val_batch_size, pin_memory=True, shuffle=True, num_workers=8, drop_last=True)
    
    def validation_step(self, batch, batch_idx):
        data = batch
        device = self.device
        
        condition_seq = awgn(data["measurements"].permute(0, 3, 1, 2).to(device), 30)
        training_seq = data["pointclouds"].to(device)
        image = data["image"].to(device)

        with torch.no_grad():
            loss = self(condition_seq, image, training_seq)


        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {}

    def on_validation_epoch_end(self, outputs=None):
        # Here, you could aggregate validation metrics across all steps and log them
        # For example, if you returned {"val_loss": val_loss} in validation_step:
        # avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # self.log('avg_val_loss', avg_val_loss)

        # Assuming you want to generate and log samples now:
        self.model.eval()
        with torch.no_grad():
            data = next(iter(self.val_dataloader()))  # Get a batch from the validation set
            self.sample(data, "test_Sample")
            
        if self.train_testing is not None:
            with torch.no_grad():
                self.sample(self.train_testing, "train_Samples")
                
        self.model.train()
        self.model.set_gradient()
        # Reuse the sample method logic here
        # Ensure you modify the method to work with a single batch from validation set
        # As an example, you might directly use parts of the sample method logic here

        # self.model.train()  # Set the model back to training mode
        
        
    def sample(self, data, name="test"):
        condition_seq = awgn(data["measurements"].permute(0, 3, 1, 2).to(self.device), 30)[:8]
        pc = data["pointclouds"][:8].to(self.device)
        image = data["image"][:8]
        
        pc_to_log = []
        
        self.model.eval() 
        gt = self.model.encode_pointcloud(pc)
        
        with torch.no_grad():
            predicted_image_latent, predicted_sdf_latent, _ = self.model.encode(condition_seq)
            predicted_images = (self.model.decode_image_latent(predicted_image_latent)+1)/2.
            
            # predicted_images = self.model.decode_image_latent(predicted_images.to(self.device)).cpu()
        
        sdfs = []
        sdf_true = []
        
        for i in range(0, len(self.model.temp), 48000):
            cur = self.model.temp[i:i+48000][None].expand(len(predicted_sdf_latent), -1, -1).to(self.device)
            sdf = self.model.decode_sdf_latent(cur, predicted_sdf_latent)
            sf_t = self.model.decode_sdf_latent(cur, gt)
            sdfs.append(sdf.cpu())
            sdf_true.append(sf_t.cpu())
            
        sdfs = torch.cat(sdfs, 1)
        sdf_true = torch.cat(sdf_true, 1)
        
        data=[]
        columns = ["Measurement", "True Images", "Predicted Images", "True Object", "Predicted Object"]
        for i, sdf in enumerate(sdfs):
            
            sdf = sdf-sdf.mean() if torch.all(sdf < 0) or torch.all(sdf > 0) else sdf
            sdf_t = sdf_true[i] - sdf_true[i].mean() if torch.all(sdf_true[i] < 0) or torch.all(sdf_true[i] > 0) else sdf_true[i]
            sdf = sdf.view(self.model.temp.grid_size, self.model.temp.grid_size, self.model.temp.grid_size).cpu().numpy()
            sdf_t = sdf_t.view(self.model.temp.grid_size, self.model.temp.grid_size, self.model.temp.grid_size).cpu().numpy()
            
            verts, faces, normals, _ = skimage.measure.marching_cubes(
            volume=sdf,
            level=0,
            allow_degenerate=False,
            spacing=(self.model.temp.voxel_size,) * 3,
            )
            
           
            # The triangles follow the left-hand rule, but we want to
            # follow the right-hand rule.
            # This syntax might seem roundabout, but we get incorrect
            # results if we do: x[:,0], x[:,1] = x[:,1], x[:,0]
            old_f1 = faces[:, 0].copy()
            faces[:, 0] = faces[:, 1]
            faces[:, 1] = old_f1
            
            verts += self.model.temp.min_coord
            
            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                normals=normals,
            )

            points, _ = trimesh.sample.sample_surface(mesh, 4096)
            
            verts, faces, normals, _ = skimage.measure.marching_cubes(
            volume=sdf_t,
            level=0,
            allow_degenerate=False,
            spacing=(self.model.temp.voxel_size,) * 3,
            )
            
            old_f1 = faces[:, 0].copy()
            faces[:, 0] = faces[:, 1]
            faces[:, 1] = old_f1
            
            verts += self.model.temp.min_coord
            
            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                normals=normals,
            )
            pc, _ = trimesh.sample.sample_surface(mesh, 4096)
            
            d = [wandb.Image(condition_seq.cpu()[i]), wandb.Image(image[i]), wandb.Image(predicted_images[i]), wandb.Object3D(pc),  wandb.Object3D(points)]
            data.append(d)
            
        return wandb.Table(columns=columns, data=data)