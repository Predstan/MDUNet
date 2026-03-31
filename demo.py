import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from deepTrainer import *
from vis import visualize_scene_360, print_images
from PIL import Image

# ── CLI arguments ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="MDUNet Demo")
parser.add_argument("--save_image", action="store_true",
                    help="Save a static PNG snapshot instead of video (use with --save_video to save both)")
parser.add_argument("--save_video", action="store_true",
                    help="Save a rotating 360° MP4 video (default behavior)")
parser.add_argument("--ckpt", type=str,
                    default="./checkpoints/mdunet_1.ckpt",
                    help="Path to MDUNet checkpoint")
parser.add_argument("--sdf_ckpt", type=str,
                    default="./checkpoints/mdunet_sdf.ckpt",
                    help="Path to SDF VAE checkpoint")
demo_args = parser.parse_args()

# Default: produce video unless --save_image is explicitly set alone
if not demo_args.save_image and not demo_args.save_video:
    demo_args.save_video = True

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = torch.load(demo_args.ckpt, map_location="cpu")
args = ckpt["hyper_parameters"]          # plain dict
args["sdf_path"] = demo_args.sdf_ckpt   # override SDF checkpoint path

model = NlosModelLightning(args)
model.load_state_dict(ckpt["state_dict"])
model = model.eval()

print("Number of Parameters: ", sum([p.numel() for p in model.parameters()]))


# ── Helper functions ──────────────────────────────────────────────────────────
def get_mesh(model, sdf):
    sdfs = []
    for i in range(0, len(model.model.temp), 48000):
        cur = model.model.temp[i:i+48000][None].expand(len(sdf), -1, -1)
        sdfss = model.model.sdf_vae.predict_occupancy(cur, sdf)
        sdfs.append(sdfss.cpu())

    sdf = torch.cat(sdfs, 1)[0]
    sdf = sdf - sdf.mean() if torch.all(sdf < 0) or torch.all(sdf > 0) else sdf
    sdf = sdf.view(
        model.model.temp.grid_size,
        model.model.temp.grid_size,
        model.model.temp.grid_size
    ).cpu().numpy()

    verts, faces, normals, _ = skimage.measure.marching_cubes(
        volume=sdf,
        level=0,
        allow_degenerate=False,
        spacing=(model.model.temp.voxel_size,) * 3,
    )
    verts += model.model.temp.min_coord

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)
    return mesh


def estimate(measurements, model):
    device = "cpu"
    if isinstance(measurements, dict):
        if isinstance(measurements["measurement"], str):
            measurement = torch.load(measurements["measurement"], map_location="cpu")
        else:
            measurement = measurements["measurement"]
    else:
        measurement = measurements

    measurement = measurement / measurement.max()
    measurement = torch.flip(measurement.to(device).permute(0, 3, 1, 2), (2,))

    im = model.model.encode(measurement)
    im, sdf = im[0], im[1]
    im = (model.model.decode_image_latent(im) + 1) / 2
    im = im[0].permute(1, 2, 0)
    mesh = get_mesh(model, sdf)

    if isinstance(measurements, dict):
        measurements["mesh"] = (np.asarray(mesh.vertices), np.asarray(mesh.faces))
        measurements["scene"] = im.cpu().numpy()
        return measurements
    return im, mesh


# ── Scenes ────────────────────────────────────────────────────────────────────

scenes = {"ball_smiles":{"measurement":"measurements/ball_on_smile.pt",
                         "occluder_size":(0.1, 0.1, .1)}, 
           "ball_complex":{"measurement":"measurements/ball_on_complex.pt",
                           "occluder_size":(0.1, 0.1, 0.1,)},
            "real_chair":{"measurement":"measurements/random_real_ball_on_chair.pt",
                          "occluder_size":(0.15, 0.02, 0.22,)},
            "random_mush": {"measurement":"measurements/random_on_mush.pt", 
                            "occluder_size":(0.15, 0.02, 0.22,)} }


# ── Run inference ─────────────────────────────────────────────────────────────
for scene in scenes:
    print(f"\n[{scene}] Running reconstruction...")
    t0 = time.time()
    ms = estimate(scenes[scene], model)
    elapsed = time.time() - t0
    print(f"  ✅ Inference done in {elapsed:.2f}s  (visualization next — may take longer)")
    scenes[scene] = ms

# ── Build the physical NLOS forward model ─────────────────────────────────────
from world_ import PinsPeck

N = M = (64, 64)
B = PinsPeck(
    hidden_x_range=[0, .408],     hidden_y_range=(0.0, 0.0),
    hidden_z_range=[0.03, 0.03+0.3085],
    visible_x_range=[0.218, 0.218+0.9], visible_y_range=(1.076, 1.0),
    visible_z_range=[0.05, 0.05+0.9],
    hidden_resolution=N[0], visible_resolution=M[0],
    sub_resolution=1, device="cpu", grid_size=[3, 3, 3]
)

def compute_min_max(pts):
    return pts.min(0), pts.max(0)

def point_as_occluder(point0, min_box, max_box):
    """Scale mesh vertices into the target bounding box (same as demo3.py)."""
    point0 = torch.Tensor(point0).clone()
    minimum = point0.min(0)[0].unsqueeze(0)
    maximum = point0.max(0)[0].unsqueeze(0)
    val_min = torch.tensor(min_box).float().unsqueeze(0)
    val_max = torch.tensor(max_box).float().unsqueeze(0)
    point0 = ((point0 - minimum) / (maximum - minimum)) * (val_max - val_min) + val_min
    return point0

# Shared coordinate bounds (hidden + visible plane)
hidden_np  = B.hidden[:, 0].cpu().numpy()   # (N, 3)
visible_np = B.visible.cpu().numpy()
min_, max_ = compute_min_max(np.concatenate([hidden_np, visible_np]))

# Target bounding box to place the reconstructed mesh in the physical scene
min__ = [0.2, 0.5, 0.02]
max__ = [0.4, 0.7, 0.22]

# ── Collect per-scene data into lists for visualize_multiple_scenes_360 ────────
all_pointclouds  = []
all_colors       = []
all_vertices     = []
all_faces        = []
all_point_sizes  = []
subtitles        = []

for scene_name, ms in scenes.items():
    verts_raw, faces_arr = ms["mesh"]
    scene_img = ms["scene"]   # (H, W, 3) numpy float

    # ── Convert scene image → per-point RGB colors (exactly like demo3.py) ────
    plt.imshow(np.clip(scene_img, 0, 1))
    plt.axis("off")
    plt.savefig("_tmp.png", format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    c = np.asarray(
        Image.open("_tmp.png").convert("RGB")
             .resize((64, 64), Image.Resampling.LANCZOS)
    )[None, :, :, :3] / 255.
    c = np.transpose(np.flip(c[0], (0, 1)), (1, 0, 2)).reshape(-1, 3)

    # ── Place mesh vertices into the physical bounding box ────────────────────
    verts_placed = point_as_occluder(verts_raw, min__, max__).numpy()

    # Normalize hidden-plane point cloud to shared coordinate bounds
    pointcloud_norm = hidden_np.copy()   # already in physical coords

    all_pointclouds.append(pointcloud_norm)
    all_colors.append(c)
    all_vertices.append(verts_placed)
    all_faces.append(faces_arr)
    all_point_sizes.append(0.03)
    subtitles.append(f"MDUNet — {scene_name}")

# ── Visualize all scenes side-by-side ─────────────────────────────────────────
from vis import visualize_multiple_scenes_360

save_as_image = not demo_args.save_video   # image if --save_video not passed

visualize_multiple_scenes_360(
    pointclouds=all_pointclouds,
    point_colors_list=all_colors,
    mesh_vertices_list=all_vertices,
    mesh_faces_list=all_faces,
    point_sizes=all_point_sizes,
    mesh_colors_list=None,
    subtitles=subtitles,
    figsize=(12, 8),
    n_cols=len(all_pointclouds),
    output_file="shape_views.mp4" if demo_args.save_video else "shape_views.png",
    images=save_as_image,
    min_coords=min_,
    max_coords=max_,
)

output = "shape_views.mp4" if demo_args.save_video else "shape_views_left.png"
print(f"\nDone! Saved: {output}")
