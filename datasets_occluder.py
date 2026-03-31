
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
import os
from torchvision import datasets, transforms
from einops import rearrange
from torchvision import transforms as T, utils
import sys
import numpy as np
import csv
from PIL import Image
import pandas
import scipy
import glob
import math
import psutil
from joblib import Parallel, delayed
# import pyvips

# from occluder.datasets.pointcloud import PointCloud
IMAGENET_DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGENET_DEFAULT_STD =  [0.26862954, 0.26130258, 0.27577711]


def normalize(s):
    return s/s.max()


def point_as_occluder(point0, min_box=None, max_box=None, x=(0.55, 0.7), y=(0.6, 0.7), z=(0.15, 0.35), flip=False):
    x_min, x_max =x; y_min, y_max =y ; z_min, z_max = z
    reshape = len(point0.shape) < 3
    if reshape:
        point0 = point0.unsqueeze(0)
    if flip:
        point0 = point0[:, :,  [0, 2, 1]].clone()
        
    minimum = point0.min(1)[0].unsqueeze(1)
    maximum = point0.max(1)[0].unsqueeze(1)
    if min_box is None:
        val_min = min_box
        val_max = max_box
        
    else:
        val_min = min_box.unsqueeze(0).to(point0)
        val_max = max_box.unsqueeze(0).to(point0)
        
    # print(point0.shape, minimum.shape, maximum.shape, val_min.shape, val_max.shape)

    point0 = ((point0 - minimum)/(maximum-minimum))*(val_max-val_min) + val_min

    if reshape:
        return point0[0]



def point_as_occluder(point0, bounding_box_min, bounding_box_max):
       
       
    # Compute the bounding box of the mesh
    mesh_min = point0.min(0)[0] #torch.min(point0, axis=0)
    mesh_max = point0.max(0)[0] #(point0, axis=0)
   
    mesh_dims = mesh_max - mesh_min
    bounding_box_dims = bounding_box_max - bounding_box_min
   
    longest_dim_index = torch.argmax(mesh_dims)
   
    ratio = mesh_dims/mesh_dims[longest_dim_index]
    # ratio[ratio<0.3] = 0.3
    rat = bounding_box_dims[longest_dim_index]*ratio
    point0 = ((point0 - mesh_min)/(mesh_max-mesh_min))*(rat) + bounding_box_min
    return point0




class Core3DDataset(Dataset):

    def __init__(self, path_to_data, augument=True,
                image_channel=3, test_size=0.01, partition="train"):


        super(Core3DDataset, self).__init__()

        self.path_to_data=path_to_data
        self.path = glob.glob(f"{path_to_data}/**/**.pt")
        
        self.path = self.path[int(len(self.path)*test_size):] if partition == "train" else self.path[:int(len(self.path)*test_size)]
        self.augument = augument
        self.image_channel=image_channel
        self.data  = None
        
        self.data = [None]*len(self.path)
        
        self.data[0] = np.load(self.path[0])
        
        def process_mat(i):
            try:
                data = torch.load(self.path[i])
                self.data[i] = data
            except:
                self.data[i] = self.data[0]
                

        Parallel(n_jobs=90, prefer="threads")(delayed(process_mat)(i) for i in range(len(self.data)))
            
        np.random.shuffle(self.data)


    def __len__(self):
        return len(self.path)


    def __getitem__(self, idx):


        data = {}
        
        # f = torch.load(self.path[idx])
        f = self.data[idx]
        image = f["latent"] #.permute(0, 3, 1, 2 )
        m = f["measurement"][0].type(torch.float32)+1e-12
        
        r = np.random.choice( [0, 1], p = [0.5, 0.5] )
        if r and self.augument:
            image = torch.flip(image, (2,))
        else:
            m = torch.flip(m, (0,))

        r = np.random.choice( [0, 1], p = [0.9, 0.1] )
        if r==1:
            m = torch.flip(m, (-1,))
            image = torch.flip(image, (1,))
            
        data["measurements"]=m
        data["image"]=image
        return data
    
    
class CoreDataset(Dataset):

    def __init__(self, path_to_data, root_to_image=None,
                image_size=64, seq_length=2048, augument=False,
                image_channel=3, partition="train", test_size=0.01,
                images_per_model=8):


        super(CoreDataset, self).__init__()

        self.path_to_data=path_to_data

        if isinstance(path_to_data, list):
            self.path = []
            for i in range(len(path_to_data)):
                self.path += glob.glob(f"{path_to_data[i]}/{partition}**.pt")

        else:
            self.path = glob.glob(f"{path_to_data}/**/**.pt")
            
        self.images_per_model=images_per_model
        self.image_size=image_size
        self.seq_length=seq_length
        np.random.seed(32)
        np.random.shuffle(self.path)
        self.path = self.path[int(len(self.path)*test_size):] if partition == "train" else self.path[:int(len(self.path)*test_size)]
        self.augument = augument
        self.image_channel=image_channel
        self.transform = transforms.Compose([transforms.Resize((256, 256),
                                        interpolation=transforms.InterpolationMode.BICUBIC,
                                      ),
                                transforms.ToTensor(),
                                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        self.root_to_image = root_to_image
        self.time = 0
        np.random.shuffle(self.path)
        # self.path=self.path[-10000:]

        self.data = [None]*len(self.path)
        
        self.data[0] = np.load(self.path[0])
        
        def process_mat(i):
            try:
                data = torch.load(self.path[i])
                self.data[i] = data
            except:
                self.data[i] = self.data[0]
                

        Parallel(n_jobs=90, prefer="threads")(delayed(process_mat)(i) for i in range(len(self.data)))
            
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data) * self.images_per_model # len(self.data[0]["measurement"]) #len(self.path) * 8 # 


    def __getitem__(self, idx):
        # self.root_to_image=None

        data = {}

        f = self.data[idx//self.images_per_model]
        # f = self.path[idx//8]
        
        # try:
        #     f = torch.load(f)
        # except:
        #     os.remove(f)
        #     f = torch.load(self.path[0])
   
        data = {}

        pc = f["pointcloud"].type(torch.float32)
        # index = torch.randint(0, 8, (1,)).item()
        index = idx%self.images_per_model
            
        m = f["measurement"][index].type(torch.float32)+1e-12
        # image = f["scene"][index]  #/255.
        # image = f["scene"][index].permute(2, 0, 1)/255.
        # image = Image.open(self.root_to_image+"/"+path).convert("RGB").resize((256, 256))
        # image = torch.Tensor(np.array(image)/255.).permute(2, 0, 1)

        image = Image.open(os.path.join(self.root_to_image, f["image_path"][index]))
        image = image.convert("RGB")
        # # # image = Image.fromarray(image)
        
        image = self.transform(image)
        
        r = np.random.choice( [0, 1], p = [0.5, 0.5] )
        if r and self.augument:
            pc = rotate_pointcloud(pc, 180, 1)
            image = torch.flip(image, (1,))
        else:
            m = torch.flip(m, (0,))

        r = np.random.choice( [0, 1], p = [0.9, 0.1] )
        if r==1:
            m = torch.flip(m, (-1,))
            image = torch.flip(image, (0,))
    
        data = {"measurements": m}
        data["image"] = image.type(torch.float32).detach()/image.max() #.permute(2, 0, 1)
        pc = point_as_occluder(pc, torch.tensor([-0.4, -0.4, -0.4]), torch.tensor([0.4, 0.4, 0.4]))
        data['pointclouds'] = pc.type(torch.float32)
        return data


def rotate_pointcloud(pointcloud, degrees, axis):
    """
    Rotate a point cloud around a specific axis by a given number of degrees using PyTorch.
    
    Args:
        pointcloud (torch.Tensor): Input point cloud of shape (N, 3), where N is the number of points.
        degrees (float): Number of degrees to rotate the point cloud.
        axis (int): Axis to rotate around (0 for X-axis, 1 for Y-axis, 2 for Z-axis).
    
    Returns:
        torch.Tensor: Rotated point cloud.
    """
    radians = math.radians(degrees)
    rotation_matrix = torch.eye(3)
    
    cos_theta = torch.cos(torch.tensor(radians))
    sin_theta = torch.sin(torch.tensor(radians))
    
    if axis == 0:
        rotation_matrix[1, 1] = cos_theta
        rotation_matrix[1, 2] = -sin_theta
        rotation_matrix[2, 1] = sin_theta
        rotation_matrix[2, 2] = cos_theta
    elif axis == 1:
        rotation_matrix[0, 0] = cos_theta
        rotation_matrix[0, 2] = sin_theta
        rotation_matrix[2, 0] = -sin_theta
        rotation_matrix[2, 2] = cos_theta
    elif axis == 2:
        rotation_matrix[0, 0] = cos_theta
        rotation_matrix[0, 1] = -sin_theta
        rotation_matrix[1, 0] = sin_theta
        rotation_matrix[1, 1] = cos_theta
    else:
        raise ValueError("Invalid axis. Must be 0, 1, or 2.")
    
    rotated_pointcloud = torch.matmul(pointcloud, rotation_matrix)
    return rotated_pointcloud


def get_data_iterator(iterable):
    """
    
    Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
        
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
            
            
# def point_as_occluder(point0, min_box=None, max_box=None, x=(0.55, 0.7), y=(0.6, 0.7), z=(0.15, 0.35), flip=False):
#     x_min, x_max =x; y_min, y_max =y ; z_min, z_max = z
#     reshape = len(point0.shape) < 3
#     if reshape:
#         point0 = point0.unsqueeze(0)
#     if flip:
#         point0 = point0[:, :,  [0, 2, 1]].clone()
        
#     minimum = point0.min(1)[0].unsqueeze(1)
#     maximum = point0.max(1)[0].unsqueeze(1)
#     if min_box is None:
#         val_min = min_box
#         val_max = max_box
        
#     else:
#         val_min = min_box.unsqueeze(0).to(point0)
#         val_max = max_box.unsqueeze(0).to(point0)
        
#     # print(point0.shape, minimum.shape, maximum.shape, val_min.shape, val_max.shape)

#     point0 = ((point0 - minimum)/(maximum-minimum))*(val_max-val_min) + val_min

#     if reshape:
#         return point0[0]

#     return point0






if __name__=="__main__":
    import time
    from PIL import Image
    from tqdm.auto import tqdm
    
    # root_to_image="/data/fraji/ImageNet-Datasets-Downloader/imagenet"
    # path = glob.glob(f"/data/fraji/imagenet/**/**.npz")
    # # # print(len(path))
    # image = "/data/fraji/ImageNet-Datasets-Downloader/imagenet/"
    # path = glob.glob(f"/data/fraji/imagenet/**/**.pt")
    # from diffusers.models import AutoencoderKL
    # device ="cuda:0"
    # print("Here")
    # path = glob.glob(f"/data/fraji/mult_image/**/**.pt")
    # # /data/fraji/mult_image/02691156/000013_1.pt
    # print(len(path))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    
    # for i, p in enumerate(path):
    #     f = torch.load(p)
    #     image = f["scene"][0].permute(0, 3, 1, 2 )
    #     image = (image.to(device)/255.)*2-1
    #     latent = vae.encode(image).latent_dist.sample().mul_(0.18215)
    #     f["latent"] = latent.detach().cpu()
    #     torch.save(f, p)
    
    train_dset = Core3DDataset("/data/fraji/mult_image")
    
    # from diffusers import AutoencoderKL

    # url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file
    # mode = AutoencoderKL.from_single_file(url).cuda()
    
    
    # import joblib
    # from tqdm.auto import tqdm

    # class ProgressParallel(joblib.Parallel):
    #     def __call__(self, *args, **kwargs):
    #         with tqdm() as self._pbar:
    #             return joblib.Parallel.__call__(self, *args, **kwargs)

    #     def print_progress(self):
    #         self._pbar.total = self.n_dispatched_tasks
    #         self._pbar.n = self.n_completed_tasks
    #         self._pbar.refresh()
    
    
    # # with tqdm(initial = 86865, total = 86865+len(path)) as pbar:
        
    # def proceed(p):
    #     tensor = torch.load(p)
    #     pa = tensor["image_path"]
        
    #     # if tensor["scene"].size(1) == 4:
    #     #     continue
        
    #     ims = [None]*8
        
    #     def process(i):
    #     # for im in pa:
    #         im = pa[i]
    #         im = Image.open(image+im).convert("RGB").resize((128, 128))
    #         # im = 2*torch.Tensor(np.array(im)/255.).permute(2, 0, 1)[None].float()-1
    #         ims[i] = torch.from_numpy(np.array(im))[None]
    #         # ims.append(im)
            
    #     Parallel(n_jobs=8, prefer="threads")(delayed(process)(i) for i in range(len(pa)))
    
    #     ims = torch.cat(ims)
    #     # lat = mode.encode(ims.detach().cuda()).latent_dist.sample().type(torch.float16).cpu()*.18
    #     tensor["scene"] = ims 
    #     torch.save(tensor, p)
            
    # ProgressParallel(n_jobs=16, prefer="threads")(delayed(proceed)(p) for p in path)
            
            # print(i)
            # break
        # torch.save(tensor, p)
    
    # print(torch.load(path[0])[0]["scene"])
    # f=u
    # # print(len(path))
    
    # # def process_mat(i):
    # #     os.remove(path[i])
        
    # # Parallel(n_jobs=55, prefer="threads")(delayed(process_mat)(i) for i, file in enumerate(path))
        
    
    # # for i, p in enumerate(path):
    # #         now= time.time()
    # #         os.remove(p)
    # #         print(i, time.time()-now)
    

    # train_dset = CoreDataset(
    #         path_to_data="/data/fraji/imagenet", root_to_image="/data/fraji/ImageNet-Datasets-Downloader/imagenet", partition="test", test_size=0.001)
    
    # # for i in range(len(train_dset)):
        
    # #     d = train_dset[i]
        
    # #     print(i, d.keys())

    print(len(train_dset))
    # i=o
    # print(train_dset.data[0]["scene"].shape)
    val_loader = DataLoader(train_dset, batch_size=2, num_workers=2)
    i=0
    for b in val_loader:
        print(i+1)
        print(b["image"].shape)
        print(b["measurements"].shape)
        # print(b["measurements"].shape)
        i+=1
        print("\n")

    
    # # print(len(train_dset.measurements))
    # # print(len(train_dset.point_clouds))
    # # print(train_dset[10].keys())
    # print(train_dset[-1]["categories"])
    # print(train_dset[-1][self.point])
    # # print(len(train_dset.cate_indexes))
