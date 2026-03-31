import torch
from deepTrainer import *
import matplotlib.pyplot as plt

def normalize(s):
    shape = s.shape
    s = torch.reshape(s, (shape[0], -1))
    s = torch.divide(s, s.max(1)[0].unsqueeze(1))
    return torch.reshape(s, shape)


# ckpt = torch.load("/Users/fadlullahraji/Desktop/Soft-shadow-diffusion/hidden/multi_modal/check/image-epoch=49-val_loss=0.18.ckpt ", map_location="cpu")
ckpt = torch.load("/Users/fadlullahraji/Desktop/Soft-shadow-diffusion/hidden/multi_modal/check/unet-epoch=04-val_loss=0.12-v1.ckpt", map_location="cpu")
args=ckpt["hyper_parameters"]
ckpt = torch.load("/Users/fadlullahraji/Desktop/Soft-shadow-diffusion/hidden/multi_modal/check/image-epoch=29-val_loss=0.40.ckpt", map_location="cpu")

model = NlosModelLightning(args)
model.load_state_dict(ckpt["state_dict"])
# model = model.load_from_checkpoint("/Users/fadlullahraji/Desktop/Hidden_3D/multi_modal/crossformer-epoch=49.ckpt")
model = model.eval()


print(sum([p.numel() for p in model.parameters()]))


def get_mesh(model, sdf):
    sdfs = [] 
    for i in range(0, len(model.model.temp), 48000):
        cur = model.model.temp[i:i+48000][None].expand(len(sdf), -1, -1)
        plane =  model.model.sdf_vae.sdf_vae.encode_to_triplane(cur)
        sdfss = model.model.sdf_vae.predict_occupancy(cur, sdf)
        sdfs.append(sdfss.cpu())

    sdf = torch.cat(sdfs, 1)[0]
    sdf = sdf-sdf.mean() if torch.all(sdf < 0) or torch.all(sdf > 0) else sdf
    sdf = sdf.view(model.model.temp.grid_size, model.model.temp.grid_size, model.model.temp.grid_size).cpu().numpy()

    verts, faces, normals, _ = skimage.measure.marching_cubes(
    volume=sdf,
    level=0,
    allow_degenerate=False,
    spacing=(model.model.temp.voxel_size,) * 3,
    )
    verts += model.model.temp.min_coord

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        normals=normals,
    )

    # Create a 180-degree rotation matrix around the X-axis
    # rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    # mesh.apply_transform(rotation_matrix)
    
    return mesh

def estimate(measurements, model):
    device="cpu"
    if isinstance(measurements, dict):
        if isinstance(measurements["measurement"], str):
            measurement = torch.load(measurements["measurement"], map_location="cpu")
        else:measurement=measurements["measurement"]
        # occluder_relative_size = measurements["occluder_size"]
    else:
        measurement=measurements


    measurement=measurement/measurement.max()
    
    measurement  = torch.flip(measurement.to(device).permute(0, 3, 1, 2), (2,))
    
    im = model.model.encode(measurement)
    im, sdf = im[0], im[1]
    im = (model.model.decode_image_latent(im)+1)/2
    im = im[0].permute(1, 2, 0)
    mesh = get_mesh(model, sdf)
    
    if isinstance(measurements, dict):
        measurements["mesh"] = (np.asarray(mesh.vertices), np.asarray(mesh.faces))
        measurements["scene"] = im.cpu().numpy()
        print(True)
        return measurements
    return im, mesh




scenes = {"ball_smiles":{"measurement":"../../measurements/ball_on_smile.pt",
                         "occluder_size":(0.1, 0.1, .1)}, 
           "ball_complex":{"measurement":"../../measurements/ball_on_complex.pt",
                           "occluder_size":(0.1, 0.1, 0.1,)},
            "real_chair":{"measurement":"../../measurements/random_real_ball_on_chair.pt",
                          "occluder_size":(0.15, 0.02, 0.22,)},
            "random_mush": {"measurement":"../../measurements/random_on_mush.pt", 
                            "occluder_size":(0.15, 0.02, 0.22,)} }


for scene in scenes:
    ms = estimate(scenes[scene], model)
    scenes[scene] = ms