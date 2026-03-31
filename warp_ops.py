import warp as wp
import numpy as np

@wp.kernel
def sum_along_dim0(input: wp.array3d(dtype=wp.uint8), output: wp.array2d(dtype=float), N: wp.int32):
    j, k = wp.tid()
    if j < input.shape[1] and k < input.shape[2]:
        sum = float(0.0)
        for i in range(N):
           sum = sum + float(input[i, j, k])
        output[j, k] = sum

@wp.kernel
def sum_along_dim1(input: wp.array3d(dtype=float), output: wp.array2d(dtype=float), M: wp.int32):
    i, k = wp.tid()
    if i < input.shape[0] and k < input.shape[2]:
        sum = float(0.0)
        for j in range(M):
            sum = sum + float(input[i, j, k])
        output[i, k] = sum

@wp.kernel
def sum_along_dim2(input: wp.array3d(dtype=wp.uint8), output: wp.array2d(dtype=float), K: wp.int32):
    i, j = wp.tid()
    if i < input.shape[0] and j < input.shape[1]:
        sum = float(0.0)
        for k in range(K):
            sum = sum + float(input[i, j, k])
        output[i, j] = sum

def warp_sum(input_array, axis):
    wp.init()
    input_wp = wp.array(input_array, dtype=wp.uint8)
    N, M, K = input_array.shape
    
    if axis == 0:
        output_wp = wp.zeros((M, K), dtype=float)
        wp.launch(kernel=sum_along_dim0, dim=(M, K), inputs=[input_wp, output_wp, wp.int32(N)])
    elif axis == 1:
        output_wp = wp.zeros((N, K), dtype=float)
        wp.launch(kernel=sum_along_dim1, dim=(N, K), inputs=[input_wp, output_wp, wp.int32(M)])
    elif axis == 2:
        output_wp = wp.zeros((N, M), dtype=float)
        wp.launch(kernel=sum_along_dim2, dim=(N, M), inputs=[input_wp, output_wp, wp.int32(K)])
    else:
        raise ValueError(f"Invalid axis {axis}, must be 0, 1, or 2")
    
    wp.synchronize()
    return output_wp.numpy()

@wp.kernel
def warp_ray_cast_and_color(
    ray_origin: wp.array(dtype=wp.vec3f),     # should be flattened
    ray_dir: wp.array(dtype=wp.vec3f),        # should be flattened
    mesh: wp.uint64,                          # Warp mesh handle
    mesh_vertices: wp.array(dtype=wp.vec3f),  # shape: (V,)
    mesh_faces: wp.array(dtype=int),          # shape: (F*3,) flattened
    uv_coords: wp.array(dtype=wp.vec2f),      # shape: (V,)
    texture: wp.array3d(dtype=wp.uint8),      # (H, W, 4) RGBA
    output_image: wp.array3d(dtype=float), # (rows, cols, 4)
    rows: wp.int32,                           # image height
    cols: wp.int32,                           # image width
    use_falloff: wp.int32                     # flag for falloff
):
    tid = wp.tid()
    
    # Convert flat index to 2D coordinates
    i = tid // cols
    j = tid % cols
    
    if i >= rows or j >= cols:
        return

    # Fetch ray from flattened arrays
    ray_idx = i * cols + j
    ray_o = ray_origin[ray_idx]
    ray_d = ray_dir[ray_idx]

    # Ray-Mesh intersection
    hit = wp.mesh_query_ray(mesh, ray_o, ray_d, 1.0e6)

    # Prepare default color
    r = wp.uint8(0)
    g = wp.uint8(0)
    b = wp.uint8(0)
    a = wp.uint8(0)

    if hit.result:
        # Intersection point
        hit_point = ray_o + ray_d * hit.t

        # Get face indices from flattened array
        face_start = hit.face * 3
        idx0 = mesh_faces[face_start]
        idx1 = mesh_faces[face_start + 1]
        idx2 = mesh_faces[face_start + 2]

        # Get vertices
        v0 = mesh_vertices[idx0]
        v1 = mesh_vertices[idx1]
        v2 = mesh_vertices[idx2]

        # Calculate falloff if enabled
        falloff_val = 1.0
        if use_falloff:
            p_x = ray_o - hit_point
            dist_sq = wp.dot(p_x, p_x)
            falloff_val = (p_x[2] * p_x[2]) / (dist_sq * dist_sq + 1.0e-6)

        # Barycentric coordinates
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = hit_point - v0

        d00 = wp.dot(v0v1, v0v1)
        d01 = wp.dot(v0v1, v0v2)
        d11 = wp.dot(v0v2, v0v2)
        d20 = wp.dot(v0p, v0v1)
        d21 = wp.dot(v0p, v0v2)
        denom = d00 * d11 - d01 * d01

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        # Interpolate UV
        uv0 = uv_coords[idx0]
        uv1 = uv_coords[idx1]
        uv2 = uv_coords[idx2]
        uv_p = u * uv0 + v * uv1 + w * uv2

        # Texture dimensions
        H = texture.shape[0]
        W = texture.shape[1]

        # Bilinear sampling coordinates
        x = uv_p[0] * (float(W) - 1.0)
        y = (1.0 - uv_p[1]) * (float(H) - 1.0)

        x_floor = wp.clamp(int(wp.floor(x)), 0, W-1)
        y_floor = wp.clamp(int(wp.floor(y)), 0, H-1)
        x_ceil = wp.clamp(int(wp.ceil(x)), 0, W-1)
        y_ceil = wp.clamp(int(wp.ceil(y)), 0, H-1)

        dx = x - float(x_floor)
        dy = y - float(y_floor)

        # Sample texture with proper indexing
        c00 = wp.vec4f(float(texture[y_floor, x_floor, 0]),
                      float(texture[y_floor, x_floor, 1]),
                      float(texture[y_floor, x_floor, 2]),
                      float(texture[y_floor, x_floor, 3]))
        c01 = wp.vec4f(float(texture[y_floor, x_ceil, 0]),
                      float(texture[y_floor, x_ceil, 1]),
                      float(texture[y_floor, x_ceil, 2]),
                      float(texture[y_floor, x_ceil, 3]))
        c10 = wp.vec4f(float(texture[y_ceil, x_floor, 0]),
                      float(texture[y_ceil, x_floor, 1]),
                      float(texture[y_ceil, x_floor, 2]),
                      float(texture[y_ceil, x_floor, 3]))
        c11 = wp.vec4f(float(texture[y_ceil, x_ceil, 0]),
                      float(texture[y_ceil, x_ceil, 1]),
                      float(texture[y_ceil, x_ceil, 2]),
                      float(texture[y_ceil, x_ceil, 3]))

        # Bilinear interpolation
        sample_color = (
            (1.0 - dx) * (1.0 - dy) * c00 +
            dx * (1.0 - dy) * c01 +
            (1.0 - dx) * dy * c10 +
            dx * dy * c11
        )

        # Apply falloff if enabled
        if use_falloff:
            sample_color = wp.vec4f(
                sample_color[0] * falloff_val,
                sample_color[1] * falloff_val,
                sample_color[2] * falloff_val,
                255.0  # Keep alpha at maximum when using falloff
            )

    # Write to 3D output array
    output_image[i, j, 0] = sample_color[0]
    output_image[i, j, 1] = sample_color[1]
    output_image[i, j, 2] = sample_color[2]
    output_image[i, j, 3] = sample_color[3]

def ray_cast_warp(scene, rays_o, rays_d, texture_image, use_falloff=False, sum_up=False):
    """
    Ray casting function using Warp.
    Args:
        rays_o: (rows, cols, 3) float32 array of ray origins
        rays_d: (rows, cols, 3) float32 array of ray directions
        scene: Trimesh object with vertices, faces, and UV coordinates
        texture_image: PIL image (RGBA or convertible to RGBA)
        use_falloff: If True, applies distance-based falloff and ignores alpha channel
    
    Returns:
        (rows, cols, 4) uint8 array containing the rendered image
    """
    wp.init()

    rows, cols = rays_o.shape[:2]
    assert rays_d.shape[:2] == (rows, cols), "rays_o and rays_d must match shape"

    # Flatten and convert input arrays
    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)
    
    rays_o_wp = wp.array(rays_o_flat, dtype=wp.vec3f)
    rays_d_wp = wp.array(rays_d_flat, dtype=wp.vec3f)

    # Convert mesh data
    vertices_wp = wp.array(scene.vertices, dtype=wp.vec3f)
    faces_flat = scene.faces.reshape(-1)  # Flatten to 1D
    faces_wp = wp.array(faces_flat, dtype=int)
    uv_wp = wp.array(scene.visual.uv, dtype=wp.vec2f)

    # Convert texture to RGBA
    texture_np = np.asarray(texture_image.convert("RGBA"))
    texture_wp = wp.array(texture_np, dtype=wp.uint8)

    # Create mesh
    mesh = wp.Mesh(
        points=vertices_wp,
        indices=faces_wp
    )

    # Allocate 3D output array
    output_wp = wp.zeros((rows, cols, 4), dtype=float)

    # Launch kernel
    wp.launch(
        kernel=warp_ray_cast_and_color,
        dim=rows * cols,  # Launch as 1D grid
        inputs=[
            rays_o_wp,
            rays_d_wp,
            mesh.id,
            vertices_wp,
            faces_wp,
            uv_wp,
            texture_wp,
            output_wp,
            wp.int32(rows),
            wp.int32(cols),
            wp.int32(1 if use_falloff else 0)
        ]
    )
    wp.synchronize()
    
    if sum_up:
        output_ = wp.zeros((rows, 4), dtype=float)
        wp.launch(kernel=sum_along_dim1, dim=(rows, 4), inputs=[output_wp, output_, wp.int32(cols)])
        wp.synchronize()
        return output_.numpy()

    # Return the 3D array directly
    return output_wp.numpy()


@wp.kernel
def warp_ray_cast(
    ray_origin: wp.array(dtype=wp.vec3f),     # should be flattened
    ray_dir: wp.array(dtype=wp.vec3f),        # should be flattened
    mesh: wp.uint64,                          # Warp mesh handle
    output_hits: wp.array(dtype=float),       # flattened array
    rows: wp.int32,                           # image height
    cols: wp.int32                            # image width
):
    tid = wp.tid()
    
    if tid >= rows * cols:
        return

    # Fetch ray from flattened arrays
    ray_o = ray_origin[tid]
    ray_d = ray_dir[tid]

    # Ray-Mesh intersection
    hit = wp.mesh_query_ray(mesh, ray_o, ray_d, 1.0e6)

    # Store hit distance or -1 if no hit
    if hit.result:
        output_hits[tid] = hit.t
    else:
        output_hits[tid] = -1.0

def ray_cast_warp(scene, rays_o, rays_d):
    """
    Simple ray casting function using Warp, returns hit distances.
    Args:
        rays_o: (rows, cols, 3) float32 array of ray origins
        rays_d: (rows, cols, 3) float32 array of ray directions
        scene: Trimesh object with vertices and faces
    
    Returns:
        (rows, cols) float array containing hit distances (-1 for misses)
    """
    wp.init()

    rows, cols = rays_o.shape[:2]
    assert rays_d.shape[:2] == (rows, cols), "rays_o and rays_d must match shape"

    # Flatten and convert input arrays
    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)
    
    rays_o_wp = wp.array(rays_o_flat, dtype=wp.vec3f)
    rays_d_wp = wp.array(rays_d_flat, dtype=wp.vec3f)

    # Convert mesh data
    vertices_wp = wp.array(scene.vertices, dtype=wp.vec3f)
    faces_flat = scene.faces.reshape(-1)  # Flatten to 1D
    faces_wp = wp.array(faces_flat, dtype=int)

    # Create mesh
    mesh = wp.Mesh(
        points=vertices_wp,
        indices=faces_wp
    )

    # Allocate flattened output array
    output_wp = wp.zeros(rows * cols, dtype=float)

    # Launch kernel
    wp.launch(
        kernel=warp_ray_cast,
        dim=rows * cols,  # Launch as 1D grid
        inputs=[
            rays_o_wp,
            rays_d_wp,
            mesh.id,
            output_wp,
            wp.int32(rows),
            wp.int32(cols)
        ]
    )
    wp.synchronize()

    # Reshape output back to 2D
    return output_wp.numpy().reshape(rows, cols)


@wp.kernel
def compute_rays(
    origins: wp.array(dtype=wp.vec3f),         # (N,) array of origin points
    destinations: wp.array(dtype=wp.vec3f),    # (M,) array of destination points
    output_origins: wp.array(dtype=wp.vec3f),  # (N*M,) flattened output origins
    output_dirs: wp.array(dtype=wp.vec3f)      # (N*M,) flattened output directions
):
    i, j = wp.tid()  # Get 2D thread index
    
    if i >= origins.shape[0] or j >= destinations.shape[0]:
        return
        
    # Get origin and destination points
    origin = origins[i]
    dest = destinations[j]
    
    # Calculate direction vector
    dir = dest - origin
    
    # Normalize
    length = wp.sqrt(wp.dot(dir, dir))
    if length > 1e-10:
        dir = dir / length
    
    # Write to outputs
    idx = i * destinations.shape[0] + j
    output_origins[idx] = origin  # Broadcast origin
    output_dirs[idx] = dir

def compute_ray_origins_and_directions(origins, destinations):
    """
    Compute broadcasted origins and normalized direction vectors using Warp.
    
    Args:
        origins: (N, 3) float32 array of origin points
        destinations: (M, 3) float32 array of destination points
        
    Returns:
        origins: (N, M, 3) float32 array of broadcasted origins
        directions: (N, M, 3) float32 array of normalized direction vectors
    """
    wp.init()
    
    N = len(origins)
    M = len(destinations)
    
    # Convert inputs to Warp arrays
    origins_wp = wp.array(origins.astype(np.float32), dtype=wp.vec3f)
    destinations_wp = wp.array(destinations.astype(np.float32), dtype=wp.vec3f)
    
    # Allocate output arrays (flattened)
    output_origins_wp = wp.zeros(N * M, dtype=wp.vec3f)
    output_dirs_wp = wp.zeros(N * M, dtype=wp.vec3f)
    
    # Launch kernel
    wp.launch(
        kernel=compute_rays,
        dim=(N, M),  # 2D launch grid
        inputs=[
            origins_wp,
            destinations_wp,
            output_origins_wp,
            output_dirs_wp
        ]
    )
    wp.synchronize()
    
    # Get results and reshape
    output_origins = output_origins_wp.numpy().reshape(N, M, 3)
    output_dirs = output_dirs_wp.numpy().reshape(N, M, 3)
    
    return output_origins, output_dirs


@wp.kernel
def matrix_vector_multiply_64(
    A: wp.array2d(dtype=wp.float64),
    x: wp.array2d(dtype=wp.float64),  # 2D RGB/Grayscale image
    y: wp.array2d(dtype=wp.float64),  # Output for each channel
    cols: wp.int32):
    
    tid = wp.tid()
    if tid < y.shape[0]:
        for c in range(x.shape[1]):  # Iterate over the RGB channels (or single channel)
            acc = wp.float64(0.0)
            for j in range(cols):
                acc += A[tid, j] * x[j, c]  # Access each channel individually
            y[tid, c] = acc


@wp.kernel
def matrix_vector_multiply(
    A: wp.array2d(dtype=float),
    x: wp.array2d(dtype=float),  # 2D RGB/Grayscale image
    y: wp.array2d(dtype=float),  # Output for each channel
    cols: wp.int32):
    
    tid = wp.tid()
    if tid < y.shape[0]:
        for c in range(x.shape[1]):  # Iterate over the RGB channels (or single channel)
            acc = float(0.0)
            for j in range(cols):
                acc += A[tid, j] * x[j, c]  # Access each channel individually
            y[tid, c] = acc

        
class ForwardModel:
    def __init__(self, measurement_dim: int, image_dim: int, dtype=wp.float64):
        self.measurement_dim = measurement_dim
        self.image_dim = image_dim
        self.dtype=dtype
        self._function = matrix_vector_multiply if dtype == float else matrix_vector_multiply_64
        
    def set_forward_model(self, A_host: np.ndarray):
        self.A = wp.array(A_host, dtype=self.dtype)
    
    def forward(self, x_host: np.ndarray) -> np.ndarray:
        x_dev = wp.array(x_host, dtype=self.dtype)
        y_dev = wp.zeros((self.measurement_dim*self.measurement_dim, x_host.shape[-1]), dtype=self.dtype)
        
        wp.launch(
            kernel=self._function,
            dim=self.measurement_dim*self.measurement_dim,
            inputs=[self.A, x_dev, y_dev, wp.int32(x_host.shape[0])]
        )
        wp.synchronize()
        
        y_dev = y_dev.numpy().reshape(self.measurement_dim, self.measurement_dim, 3)
        return y_dev/y_dev.max()

if __name__=="__main__":
    import trimesh
    from PIL import Image
    import numpy as np

    import glob
    import os
    files = glob.glob("/data/fraji/objaverse/glbs/000-001/**.glb")
    # file = "/Users/fadlullahraji/Desktop/sandbox-TERI-ML/02773838/22b7d6fa819d62aefc69b7db9c6d5ad9/models/model_normalized.obj"
    idxs = np.random.choice(range(len(files)))
    locations = [[1.1, 0.3, 0], [1.1, 0.6, 0], [1.5, 1.0, 0], [2.337, 0.8, 0], [2.337, 1.0, 0], [2.5, 1.0, 0]]
    # idx=24
    # print(env.room_center)
    # # f = env.room_center
    # # f[1]=0.8

    # for l in locations:
    #     idx = np.random.choice(range(len(files)))
    #     env.add_object(trimesh.load(files[idx], force="mesh"), location=l, scale=1.3)
        
    # env.show(follow_object=False) 
    # im, rt = env.capture(resolution=512, follow_object=False, fov=32)
    # Image.fromarray(im).save(f"image.png")
    # # env.add_object(trimesh.load(files[idx], force="mesh"), scale=1.3)
    # # env.add_object(trimesh.load(files[idxs], force="mesh"),location=f, scale=1.2)
    # env.show()
    mesh = trimesh.load(files[idxs], force="mesh")
        
    # Convert the mesh material to simple, if needed
    try:
        mesh.visual.material = mesh.visual.material.to_simple()
    except:
        pass

    from utils import *
    env= Environment(
            room_x=(0.0, 1+2.6752),
            room_y=(0.0, 2.6752),
            room_z=(0.0, 2.6752),
            
            door_x=(1, 1+2.6752), # Then the door witdh will be 2.6752 * overhang width
            door_y=(0.0, 0.02),
            door_z=(0.0, 2.6752-0.6256), # h=0.625*overhang width ( Then the door ranges from 0 to room height - overhang width)
            
            overhang_x=(0, 1), # Say overhang is 1 according to the nature paper
            overhang_y=(-1, 0), # Say overhang is the 1 according to 
            
            
            light_x = (2.6+1, 2.6752+1), 
            light_y = [0.5, 1.+.25],
            light_z = [0.3, 1.3],
            
            add_frames=True,
            wall_thickness=0.02,
            camera_pixel=224,
            num_points=2048,
            scene_pixel=256,
            light_inntensity=5,
            add_light=False,
            number_of_true_camera=4,
        )


    from PIL import Image
    mesh = env.add_object(mesh, locations[3], scale=1.3)
    import time
    now = time.time()
    im, RT, rays = env.capture(resolution=512)
    res_pil = Image.fromarray(im, mode="RGBA").save("o3d.png")
    print("o3d", time.time()-now)

    rays = rays.numpy()

    rays_o = rays[:, :, :3]
    rays_d = rays[:, :, 3:]

    now = time.time()
    texture_image = mesh.visual.material.image
    result_image = ray_cast_warp(mesh, rays_o, rays_d, texture_image)
    res_pil = Image.fromarray(result_image, mode="RGBA").save("m.png")
    print("cuda", time.time()-now)
    # res_pil.show()

