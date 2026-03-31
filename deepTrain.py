from deepTrainer import *
import random

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.set_float32_matmul_precision('medium')
parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None)     
parser.add_argument('--path_to_data', type=str, default="/data/fraji/imagenet")                 
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--unet', type=int, default=0)
parser.add_argument('--num_devices', type=int, default=1)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--warmup_epoch', type=int, default=1)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--results_folder', type=str, default="./check")  
parser.add_argument("--super_resolve", type=int, default=False)
parser.add_argument("--fine_tune_sdf", type=int, default=False)
parser.add_argument("--fine_tune_image", type=int, default=False)
parser.add_argument('--sdf_path', type=str, default="./checkpoints/sdf.ckpt") 



def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed)
    
args = parser.parse_args()
seed_all(args.seed)

# trainer= Trainer( device=argss.device, sdfModel=model, args=args, z_dim=argss.z_dim)
# class ValEveryNSteps(pl.Callback):                                                                                                                                                                                 
#     def __init__(self, every_n_steps):                                                                                                                                                                             
#         self.last_run = None                                                                                                                                                                                      
#         self.every_n_steps = every_n_steps                                                                                                                                                                         
                                                                                                                                                                                                                   
#     def on_batch_end(self, trainer, pl_module):                                                                                                                                                                    
#         # Prevent Running validation many times in gradient accumulation                                                                                                                                           
#         if trainer.global_step == self.last_run:                                                                                                                                                                   
#             return                                                                                                                                                                                                 
#         else:                                                                                                                                                                                                      
#             self.last_run = None                                                                                                                                                                                   
#         if trainer.global_step % self.every_n_steps == 0 and trainer.global_step != 0:                                                                                                                             
#             trainer.training = False                                                                                                                                                                               
#             stage = trainer.state.stage                                                                                                                                                                            
#             trainer.state.stage = RunningStage.VALIDATING                                                                                                                                                          
#             trainer._run_evaluate()                                                                                                                                                                                
#             trainer.state.stage = stage                                                                                                                                                                            
#             trainer.training = True                                                                                                                                                                                
#             trainer.logger_connector._epoch_end_reached = False                                                                                                                                                    
#             self.last_run = trainer.global_step


def main(args):
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.strategies import DDPStrategy
    # Initialize WandB logging
    name="unet" if args.unet else "crossformers"
    name=name if not args.super_resolve else "super_resolve"
    wandb_logger = WandbLogger(project="OccluderGen", name=name)
    
    model = NlosModelLightning(args)
    
    ckpt = torch.load("/data/fraji/Hidden_3D/multi_modal/check/unet-epoch=04-val_loss=0.12-v1.ckpt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    
    model.args.train_batch_size=args.train_batch_size

    # DDPStrategy(process_group_backend="ddp"),
    # Setup your ModelCheckpoint callback as previously described
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.results_folder,
        filename=name+'-{global_step:02d}-{val_loss:.2f}',
        monitor='epoch',
        save_top_k=3,
        mode='max', 
    )
    
    # Configure the Trainer to use 3 GPUs
    trainer = Trainer(
        max_steps=1000000,
        max_epochs=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator='gpu',  # Specify 'gpu' as the accelerator
        devices=args.num_devices,  # Use 3 GPUs
        check_val_every_n_epoch=5,
        strategy=DDPStrategy(find_unused_parameters=True),
        gradient_clip_val=0.1,
        num_nodes=1,
        precision="bf16-mixed",
    )
    # Train the model
    trainer.fit(model)
    # trainer.fit(model, ckpt_path="/data/fraji/Hidden_3D/multi_modal/check/super_resolve-epoch=69-val_loss=0.20.ckpt")
    
if __name__ == "__main__":
    # if args.unet:
    #     trainer = Trainer(args=args, device=args.device)
    #     trainer.train(2000000)
    # else:
    main(args)
    