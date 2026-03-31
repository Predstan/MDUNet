
from ema_pytorch import EMA
from utils import *
from datasets_occluder import CoreDataset, DataLoader
# from pytorch3d.loss import chamfer_distance
from model import *
from accelerate import Accelerator
import argparse
import wandb
from pathlib import Path
from torch.optim import Adam
import random
from tqdm.auto import tqdm
import trimesh
import skimage
import math

from accelerate import DistributedDataParallelKwargs



def cycle(dl):
    while True:
        for data in dl:
            yield data
            
            
def exists(x):
    return x is not None

# trainer class
class OccluderTrainer(object):
    def __init__(
        self,
        *,
        args=None,
        experiment_name="OccluderGen",
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 1000000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 50000,
        results_folder = './check',
        amp = False,
        fp16 = True,
        logger = "wandb"
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        # accelerator
        self.accelerator = Accelerator(
            mixed_precision = 'fp16' if fp16 else 'no',
            log_with=logger,
            kwargs_handlers=[ddp_kwargs]
        )

        self.accelerator.native_amp = amp
        
        model = NlosModelLightning(args)
    
        ckpt = torch.load("/data/fraji/Hidden_3D/multi_modal/check/diffusion-3.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        
        if args.super_resolve:
            model = Model3D(args, model.model)
            # ckpt = torch.load("/data/fraji/Hidden_3D/multi_modal/check/diffusion-1.pt", map_location="cpu")
            # model.load_state_dict(ckpt["model"])
        
        self.model = model
        self.loss_meter = AverageMeter()
        self.args=args
        self.ema_update_every=ema_update_every
    
   
        self.save_and_sample_every = save_and_sample_every
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.logger = logger
 

        self.train_ds, dl = model.train_dataloader()
        print(f"Number of Training Examples in {experiment_name}: ", len(self.train_ds))
    

        if self.accelerator.is_main_process:
            if logger == "wandb":
                self.accelerator.init_trackers(project_name="OccluderGen",)
                self.accelerator.trackers[0].run.name = experiment_name

            self.val_ds, test_dl = model.val_dataloader()
            
            print(f"Number of Validation Examples in {experiment_name}: ", len(self.val_ds))
            self.test_dl = cycle(test_dl)
        
        # optimizer
        params_to_update = [param for param in self.model.parameters() if param.requires_grad]
        self.opt = Adam(params_to_update, lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.dl = cycle(dl)


    @property
    def device(self):
        return self.accelerator.device
    

    def save(self, milestone, val_data=None):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            "args":self.args,
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        # if data is not None:
        #     torch.save(val_data, str(self.results_folder / f'val-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location="cpu")
        
        model =  self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])
        
        opt = self.accelerator.unwrap_model(self.opt)
        opt.load_state_dict(data["opt"])
        
    
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

        self.model, self.opt = self.accelerator.prepare(model , opt)

   
    def log_samples(self, image, true_sample, recons):
            pass

   
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        print("Device Here", device)

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.
                
                for _ in range(self.gradient_accumulate_every):
                    
                    data = next(self.dl)
        
                    # r = np.random.choice([0, 1], p=[0.5, 0.5])
                    # condition_seq = awgn(data["measurements"].permute(0, 3, 1, 2).to(device), 15, 100) if r else sbr(data["measurements"].permute(0, 3, 1, 2).to(device), 15, 100)
                    # image = data["image"].to(device)
                    
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                   
                    self.accelerator.backward(loss)
                
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                pbar.set_description(f'loss: {total_loss:.4f}')
                

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
               
                accelerator.wait_for_everyone()
                

                
                if accelerator.is_main_process:
                    
                    if self.step % self.ema_update_every == 0:
                 
                        self.ema.update()

                    self.accelerator.log({'train_loss': total_loss},  step=self.step)
                    # print("Exactle At This Point")
                   
                    if self.step % self.save_and_sample_every == 0:
                        
                        self.ema.ema_model.eval()
                        data = next(self.test_dl)
                        milestone = self.step // self.save_and_sample_every
                        with torch.no_grad():
                            mode = accelerator.unwrap_model(self.ema.ema_model).to(device)
                            table = mode.sample(data)
                            self.accelerator.log({"test": table},  step=self.step)
                            
                                
                        self.save(milestone, val_data=None)

                self.step += 1
                pbar.update(1)

        accelerator.print('training complete')


if __name__ == "__main__":
    
    # Datasets and loaders
    from deepTrainer import *
    import random

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # torch.set_float32_matmul_precision('medium')
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--resume', type=str, default=None)     
    # parser.add_argument('--path_to_data', type=str, default="/data/fraji/imagenet")                 
    # parser.add_argument('--train_batch_size', type=int, default=32)
    # parser.add_argument('--unet', type=int, default=0)
    # parser.add_argument('--num_devices', type=int, default=1)
    # parser.add_argument('--val_batch_size', type=int, default=64)
    # parser.add_argument('--epochs', type=int, default=500)
    # parser.add_argument('--warmup_epoch', type=int, default=1)
    # parser.add_argument('--seed', type=int, default=2024)
    # parser.add_argument('--results_folder', type=str, default="./check") 
    # parser.add_argument('--val_freq', type=int, default=50000)
    # parser.add_argument("--super_resolve", type=int, default=False)
    # parser.add_argument("--fine_tune_sdf", type=int, default=False)
    # parser.add_argument("--fine_tune_image", type=int, default=False)
    # parser.add_argument('--sdf_path', type=str, default="/data/fraji/Estimate_3D/implicit/checkpoints/sdf_vae/model-epoch=779-val_loss=0.00.ckpt") 
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)     
    parser.add_argument('--path_to_data', type=str, default="/data/fraji/mult_image")                 
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--unet', type=int, default=0)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--warmup_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--results_folder', type=str, default="./check") 
    parser.add_argument('--val_freq', type=int, default=20000)
    parser.add_argument("--super_resolve", type=int, default=False)
    parser.add_argument("--fine_tune_sdf", type=int, default=False)
    parser.add_argument("--fine_tune_image", type=int, default=False)
    parser.add_argument('--sdf_path', type=str, default="/data/fraji/Estimate_3D/implicit/checkpoints/sdf_vae/model-epoch=779-val_loss=0.00.ckpt") 
    
    def seed_all(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    args = parser.parse_args()
    seed_all(args.seed)
    
    Trainer = OccluderTrainer(
        args=args,
        gradient_accumulate_every =1,
        train_lr = 1e-4,
        train_num_steps = 5000000,
        ema_update_every = 100,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = args.val_freq,
        results_folder = args.results_folder,
        amp = False,
        fp16 = True,
        logger = "wandb")
    
    if args.resume:
        print("Resuming from Trained Diffusion")
        Trainer.load(int(args.resume.split("-")[-1].split(".")[0]))
        # Trainer.step = data["step"]
    Trainer.train()

