from crossformer import CrossFormer
from sdf_vaes import *
from utils import Query
import torch
from lpips import LPIPS
import torch
from world_ import PinsPeck,rearrange 
from einops import rearrange, reduce
from unet import UNet
# import lpips as perceptual


def prepare_coordinates():

    grid_size =  [32, 16, 32] #Define Occluder Discritization
    B = PinsPeck(hidden_x_range=[0, .408], hidden_y_range=(0.0, 0.0), hidden_z_range=[0.03, 0.03+0.3085],
                visible_x_range=[0.218, 0.218+0.9], visible_y_range=(1.0, 1.0), visible_z_range=[0.05, 0.05+0.9],
                hidden_resolution=128, visible_resolution=128, sub_resolution=1, device="cpu", grid_size=grid_size)
    
    # print(B.visible.shape, B.hidden.shape)
    # rays = rearrange(B.visible, '(h w) c -> w h c', w=128, h=128)
    # rays = positional_encoding(rays)
    rays_d = rearrange(B.visible, '(h w) c -> w h c', w=128, h=128)
    rays_o = rearrange(B.hidden, '(h w) c -> w h c', w=128, h=128) 
    rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) 
    return rays_plucker


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
  r"""Apply positional encoding to the input.

  Args:
    tensor (torch.Tensor): Input tensor to be positionally encoded.
    num_encoding_functions (optional, int): Number of encoding functions used to
        compute a positional encoding (default: 6).
    include_input (optional, bool): Whether or not to include the input in the
        computed positional encoding (default: True).
    log_sampling (optional, bool): Sample logarithmically in frequency space, as
        opposed to linearly (default: True).
  
  Returns:
    (torch.Tensor): Positional encoding of the input tensor.
  """
  # TESTED
  # Trivially, the input tensor is added to the positional encoding.
  encoding = [tensor] if include_input else []
  # Now, encode the input using a set of high-frequency functions and append the
  # resulting values to the encoding.
  frequency_bands = None
  if log_sampling:
      frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
  else:
      frequency_bands = torch.linspace(
          2.0 ** 0.0,
          2.0 ** (num_encoding_functions - 1),
          num_encoding_functions,
          dtype=tensor.dtype,
          device=tensor.device,
      )

  for freq in frequency_bands:
      for func in [torch.sin, torch.cos]:
          encoding.append(func(tensor * freq))

  # Special case, for no positional encoding
  if len(encoding) == 1:
      return encoding[0]
  else:
      return torch.cat(encoding, dim=-1)



class sdf_vae:
    
    def __init__(self, sdf_path):
            
        ckpt = torch.load(sdf_path, map_location="cpu")
        self.sdf_vae = VAEPointCloudSDFModel(ckpt["hyper_parameters"])
        self.sdf_vae.load_state_dict(ckpt["state_dict"])
        self.sdf_vae.requires_grad_(False)
        self.sdf_vae.eval()
        self.count = 0
        
        
    def encode(self, x):
        if self.count ==0:
            self.sdf_vae.to(x.device)
            self.sdf_vae.eval()
            self.count+=1
        with torch.no_grad():
            return self.sdf_vae.encode_point_clouds(x).detach()
        
    def predict_occupancy(self, points, latent):
        with torch.no_grad():
            return self.sdf_vae.predict_occupancy(points, latent).detach()
        
        
class image_vae(nn.Module):
    
    def __init__(self,):
        super(image_vae, self).__init__()
        
        # url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be a local file
        # mode = AutoencoderKL.from_single_file(url)
        
        # mode = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32)
        mode = VQModel.from_pretrained("CompVis/ldm-super-resolution-4x-openimages", subfolder="vqvae")
        self.image_vae = mode
        self.image_vae.requires_grad_(False)
        self.image_vae.eval()
        
    def encode(self, x):
        self.image_vae.eval()
        with torch.no_grad():
            return self.image_vae.encode(x).latents*0.2
        
    def decode(self, latent):
        self.image_vae.eval()
        with torch.no_grad():
            return self.image_vae.decode((latent/0.2)).sample 
        
        
class lpips:
    
    def __init__(self):
        
        self.lpips = LPIPS()
        self.lpips.requires_grad_(False)
        self.count=0
        
        
    def forward(self, x, y):
        
        if self.count ==0:
            self.lpips.to(x.device)
            self.lpips.eval()
            self.count+=1
            
        return self.lpips(x, y)
        
        
    
class NlosModel(nn.Module):
    def __init__(
        self,
        sdf_path,
        unet=True,
        fine_tune_image=False,
        fine_tune_sdf=False,
    ):
        super(NlosModel, self).__init__()
        
        
        # self.image_vae = image_vae() #AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16)
        self.sdf_vae = sdf_vae(sdf_path)
        self.fine_tune_image = fine_tune_image
        self.fine_tune_sdf = fine_tune_sdf
        self.lpips = lpips() #.from_pretrained()
        # self.lpips.requires_grad_(False)
        # self.image_vae.requires_grad_(False)
        if unet:
            # self.nlos = UNetModel(image_size=128, in_channels=3, sdf_out_channels=16, image_out_channels=3, model_channels=128, num_res_blocks=1, attention_resolutions=[8, 4, 2])
            self.nlos = UNet()
        else:
            self.nlos = ViT()
        self.temp = Query(grid_size=64, device="cpu")
        self.set_gradient()

    def encode_image(self, image):
        return 2*image-1
    
    def decode_image_latent(self, latent):
        return latent
    
    def encode_pointcloud(self, pc):
        return self.sdf_vae.encode(pc)
   
    def decode_sdf_latent(self, points, latent):
        return self.sdf_vae.predict_occupancy(points, latent)
        
    def encode(self, measurement, image_only=False, sdf_only=False):
        # ray_embedding = self.rays_embedding[None].expand(len(measurement), -1, -1, -1).to(measurement.device)
        # measurement = torch.cat([measurement, ray_embedding], 1)
        return self.nlos(measurement*2-1, image_only, sdf_only)
    
    def forward(self, measurement, image, pointcloud):

        # with torch.autocast(enabled=False, device_type="cuda"):
        image_latent = self.encode_image(image)
        sdf_latent = self.encode_pointcloud(pointcloud)

        predicted_image_latent, predicted_sdf_latent, hidden_state = self.encode(measurement, self.fine_tune_image, self.fine_tune_sdf)
      
        if self.fine_tune_image:
            image_loss = torch.nn.functional.l1_loss(predicted_image_latent, image_latent)
            image_loss = (image_loss + 0.2 * self.lpips.forward(predicted_image_latent, image_latent)).mean()
            sdf_loss = 0
            
        elif self.fine_tune_sdf:
            sdf_loss = torch.nn.functional.l1_loss(predicted_sdf_latent, sdf_latent, reduction="sum")/len(measurement)#
            image_loss=0
            
        else:
            image_loss = torch.nn.functional.l1_loss(predicted_image_latent, image_latent)
            image_loss = (image_loss + 0.2 * self.lpips.forward(predicted_image_latent, image_latent)).mean()
            sdf_loss = torch.nn.functional.l1_loss(predicted_sdf_latent, sdf_latent)

        return image_loss + sdf_loss 
            

    def set_gradient(self, fine_tune_image=False, fine_tune_sdf=False):
        if self.fine_tune_image or fine_tune_image:
            self.nlos.requires_grad_(False)
            self.nlos.image_up_blocks.requires_grad_(True)
            self.nlos.norm_image_out.requires_grad_(True)
            self.nlos.conv_image_out.requires_grad_(True)
            
        elif self.fine_tune_sdf:
            self.nlos.requires_grad_(False)
            self.nlos.sdf_up_blocks.requires_grad_(True)
            self.nlos.norm_sdf_out.requires_grad_(True)
            self.nlos.conv_sdf_out.requires_grad_(True)
        
        