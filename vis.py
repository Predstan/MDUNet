import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation
from tqdm import tqdm
import os
import trimesh
import pymeshlab

def decimate_mesh(vertices: np.ndarray, faces: np.ndarray, reduction_ratio: float = 0.5):
    """
    Decimates a 3D mesh using PyMeshLab.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        faces (np.ndarray): (M, 3) array of face indices.
        reduction_ratio (float): Fraction of original faces to retain (default 0.5 means 50% reduction).

    Returns:
        np.ndarray: Decimated vertices (N', 3).
        np.ndarray: Decimated faces (M', 3).
    """
    # Convert NumPy arrays to PyMeshLab MeshSet
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms.add_mesh(mesh)
    
    print(ms.print_filter_list())

    # Compute target number of faces
    target_faces = max(10, int(len(faces) * reduction_ratio))  # Ensure a minimum face count

    # Apply quadric decimation
    ms.apply_filter("simplification_quadric_edge_collapse_decimation", targetfacenum=target_faces)

    # Retrieve the simplified mesh
    simplified_mesh = ms.current_mesh()

    # Extract new vertices and faces
    new_vertices = simplified_mesh.vertex_matrix()
    new_faces = simplified_mesh.face_matrix()

    return new_vertices, new_faces

def visualize_scene_360(pointcloud=None, point_colors=None, point_size=0.1,
                        mesh_vertices=None, mesh_faces=None, mesh_colors=None,
                        thresh=0.1, fps=30, duration=3,mysubtitle=None,
                        figsize=(12, 8), forward_backward=2,
                        dpi=100, output_file='scene_360.mp4', images=True):
    import os
    # Ensure even dimensions for H.264 compatibility
    width_pixels = int(figsize[0] * dpi)
    height_pixels = int(figsize[1] * dpi)
    width_pixels = (width_pixels // 2) * 2
    height_pixels = (height_pixels // 2) * 2
    
    # Create figure
    fig = plt.figure(figsize=(width_pixels/dpi, height_pixels/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    # fig.patch.set_facecolor('black')
    fig.patch.set_facecolor('white')
    # fig.patch.set_facecolor('black')
    
    
    def setup_scene():
        if point_colors is not None and np.any(point_colors < 0):
            # point_colors -= point_colors.min()
            point_colors[point_colors<0] = 0
        # Plot pointcloud if provided
        if pointcloud is not None and point_colors is not None:
            # Filter points based on threshold
            if point_colors.ndim == 1:
                mask = point_colors > thresh
            else:
                # For RGB colors, use mean intensity
                mask = np.mean(point_colors, axis=1) > thresh
            
            filtered_points = pointcloud[mask]
            filtered_colors = point_colors[mask]
            
            # Normalize colors if needed
            if filtered_colors.ndim == 1:
                normalized_colors = plt.cm.viridis((filtered_colors / np.max(filtered_colors))*2.5)[:, :3]
            else:
                normalized_colors = filtered_colors.copy()
                if normalized_colors.max() > 1:
                    normalized_colors = normalized_colors / 255.0
            
            # if invert_color:
            #     normalized_colors = 1 - normalized_colors
            
            ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2],
                      c=normalized_colors, s=1000*point_size, alpha=0.6)
        
        # Plot mesh if provided
        if mesh_vertices is not None and mesh_faces is not None:    
            # mesh_triangles = mesh_vertices[mesh_faces]
            mesh = []
            for face in mesh_faces:
                vertices_face = mesh_vertices[face]
                mesh.append(vertices_face)
            mesh_collection = Poly3DCollection(mesh,
                                    facecolors='gray',
                                    alpha=0.6,
                                    edgecolor='white',
                                    linewidth=1.,
                                    antialiased=True,
                                    shade=True)
            ax.add_collection3d(mesh_collection)
        
        
        max_coords = np.array([1.1, 1.1, 1.1])
        min_coords = -np.array([1.1, 1.1, 1.1])
        # # Define and plot bounding box
        if pointcloud is not None and mesh_vertices is not None:
            # Compute the min and max across both arrays
            min_coords = np.minimum(np.min(pointcloud, axis=0), np.min(mesh_vertices, axis=0)) - 0.1
            max_coords = np.maximum(np.max(pointcloud, axis=0), np.max(mesh_vertices, axis=0)) + 0.1
        elif pointcloud is not None:
            min_coords = np.min(pointcloud, axis=0)
            max_coords = np.max(pointcloud, axis=0) * 1.1
        elif mesh_vertices is not None:
            min_coords = np.min(mesh_vertices, axis=0)
            max_coords = np.max(mesh_vertices, axis=0) * 1.1
      
        
    
        # Create bounding box vertices
        bbox = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]]
        ])
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        for edge in edges:
            ax.plot([bbox[edge[0], 0], bbox[edge[1], 0]],
                   [bbox[edge[0], 1], bbox[edge[1], 1]],
                   [bbox[edge[0], 2], bbox[edge[1], 2]],
                   color=[1, 0, 0, 0.8], linewidth=4)
        
        # Style adjustments
        ax.set_facecolor('white')
        ax.grid(False)
        ax.tick_params(axis='both', which='both',
                      left=False, right=False, bottom=False, top=False,
                      labelleft=False, labelbottom=False)
        
        # # Add axis labels and arrows
        # center = (max_coords + min_coords) / 2
        # ax.text(center[0], min_coords[1], max_coords[2], 'x', color='white', fontsize=20)
        # ax.text(min_coords[0], center[1], max_coords[2], 'y', color='white', fontsize=20)
        # ax.text(min_coords[0], min_coords[1], center[2], 'z', color='white', fontsize=20)
        
        # # Add scale bar
        # scale = np.max(max_coords - min_coords) / 5
        # ax.text(center[0], min_coords[1], min_coords[2], 
        #        f"{scale:.1f} m", color='white', fontsize=10, 
        #        ha='center', va='bottom')
        
        # # Add coordinate arrows
        # arrow_length = scale / 2
        # ax.quiver(min_coords[0], min_coords[1], min_coords[2],
        #          arrow_length, 0, 0, color='red', arrow_length_ratio=0.1)
        # ax.quiver(min_coords[0], min_coords[1], min_coords[2],
        #          0, arrow_length, 0, color='green', arrow_length_ratio=0.1)
        # ax.quiver(min_coords[0], min_coords[1], min_coords[2],
        #          0, 0, arrow_length, color='blue', arrow_length_ratio=0.1)
        
        if mysubtitle:
            plt.suptitle(mysubtitle)
        
        ax.zaxis.set_pane_color((0.2, 0.2, 0.2, 1))
        ax.xaxis.set_pane_color((0, 0, 0, 1))
        ax.yaxis.set_pane_color((0, 0, 0, 1))
        
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(min_coords[0], max_coords[0])
        ax.set_ylim(min_coords[1], max_coords[1])
        ax.set_zlim(min_coords[2], max_coords[2])
        ax.set_facecolor('white')
        
        plt.tight_layout()

    if images:
        views = [
            ('left', 20, 0),
            # ('front', 20, 90),
            # ('right', 20, 130)
        ]
        fig = plt.figure(figsize=(width_pixels/dpi, height_pixels/dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        # fig.patch.set_facecolor('black')
        setup_scene()
        
        for view_name, elev, azim in views:
            
            ax.view_init(elev=elev, azim=azim)
            
            output_image = f"{os.path.splitext(output_file)[0]}_{view_name}.png"
            plt.savefig(output_image)
            # plt.close()
            
            if os.path.exists(output_image):
                print(f"Image created successfully: {output_image}")
                print(f"File size: {os.path.getsize(output_image)} bytes")
            else:
                print(f"Error: Image file {output_image} was not created")
                
            return output_image
    
    else:
        n_frames = fps * duration
        base_angles = np.linspace(0, 270, n_frames, endpoint=False)
        angles = list(base_angles)

        setup_scene()

        # Create temporary directory for frames
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            print("Generating frames...")
            # First generate the base frames
            base_frames = []
            for i, angle in enumerate(tqdm(angles)):
                ax.view_init(elev=20, azim=angle)
                frame_path = os.path.join(tmpdir, f'frame_{i:04d}.png')
                plt.savefig(frame_path)
                base_frames.append(frame_path)
            
            # Now create and save the extended frames
            frame_count = len(base_frames)
            for i in range(1, forward_backward):
                if i % 2 == 0:
                    # Save forward frames
                    for j, frame in enumerate(base_frames):
                        new_frame_path = os.path.join(tmpdir, f'frame_{(frame_count + j-1):04d}.png')
                        os.system(f'cp "{frame}" "{new_frame_path}"')
                        frame_count += 1
                    
                else:
                    # Save reversed frames
                    for j, frame in enumerate(reversed(base_frames)):
                        new_frame_path = os.path.join(tmpdir, f'frame_{(frame_count + j-1):04d}.png')
                        os.system(f'cp "{frame}" "{new_frame_path}"')
                        frame_count += 1
                    
            
            print(f"Saving video to {output_file}...")
            import subprocess
            # Use ffmpeg with the total number of frames
            cmd = ['ffmpeg', '-y', '-framerate', str(fps), 
                '-i', os.path.join(tmpdir, 'frame_%04d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-b:v', '2000k', output_file]
            subprocess.run(cmd)

        plt.close()

        if os.path.exists(output_file):
            print(f"Video created successfully: {output_file}")
            print(f"File size: {os.path.getsize(output_file)} bytes")
        else:
            print("Error: Video file was not created")
            
        return output_file
                    


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

def visualize_multiple_scenes_360(pointclouds=None, point_colors_list=None, point_sizes=None,
                           mesh_vertices_list=None, mesh_faces_list=None, mesh_colors_list=None,
                           thresholds=None, fps=30, duration=3, subtitles=None,
                           figsize=(20, 12), forward_backward=2, n_cols=2,
                           dpi=100, output_file='multi_scene_360.mp4', images=True, 
                           min_coords=None, max_coords=None, ):
    # Ensure even dimensions for H.264 compatibility
    width_pixels = int(figsize[0] * dpi)
    height_pixels = int(figsize[1] * dpi)
    width_pixels = (width_pixels // 2) * 2
    height_pixels = (height_pixels // 2) * 2
    
    # Determine number of scenes and validate inputs
    n_scenes = 0
    if pointclouds is not None:
        n_scenes = max(n_scenes, len(pointclouds))
    if mesh_vertices_list is not None:
        n_scenes = max(n_scenes, len(mesh_vertices_list))
    
    if n_scenes == 0:
        raise ValueError("At least one pointcloud or mesh must be provided")
    
    # Normalize and validate inputs
    if pointclouds is None:
        pointclouds = [None] * n_scenes
    if point_colors_list is None:
        point_colors_list = [None] * n_scenes
    
    if point_sizes is None:
        point_sizes = [0.01] * n_scenes
    elif isinstance(point_sizes, (int, float)):
        point_sizes = [point_sizes] * n_scenes
    
    if mesh_vertices_list is None:
        mesh_vertices_list = [None] * n_scenes
    if mesh_faces_list is None:
        mesh_faces_list = [None] * n_scenes
    if mesh_colors_list is None:
        mesh_colors_list = ['gray'] * n_scenes
    
    if thresholds is None:
        thresholds = [0.1] * n_scenes
    elif isinstance(thresholds, (int, float)):
        thresholds = [thresholds] * n_scenes
    
    if subtitles is None:
        subtitles = [None] * n_scenes
    
    # Calculate grid layout
    n_rows = int(np.ceil(n_scenes / n_cols))
    
    # Create figure
    fig = plt.figure(figsize=(width_pixels/dpi, height_pixels/dpi), dpi=dpi)
    fig.patch.set_facecolor('white')

    # Create axes for each scene
    axes = []
    for i in range(n_scenes):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        axes.append(ax)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    def setup_scenes(min_coords=min_coords, max_coords=max_coords):
        for i in range(n_scenes):
            ax = axes[i]
            pointcloud = pointclouds[i]
            point_colors = point_colors_list[i]
            point_size = point_sizes[i]
            mesh_vertices = mesh_vertices_list[i]
            mesh_faces = mesh_faces_list[i]
            mesh_color = mesh_colors_list[i]
            thresh = thresholds[i]
            subtitle = subtitles[i]
            
            # Process point colors if needed
            if point_colors is not None and np.any(point_colors < 0):
                point_colors[point_colors < 0] = 0
                
            # Plot pointcloud if provided
            if pointcloud is not None and point_colors is not None:
                # Filter points based on threshold
                if point_colors.ndim == 1:
                    mask = point_colors > thresh
                else:
                    # For RGB colors, use mean intensity
                    mask = np.mean(point_colors, axis=1) > thresh
                
                filtered_points = pointcloud[mask]
                filtered_colors = point_colors[mask]
                
                # Normalize colors if needed
                if filtered_colors.ndim == 1:
                    normalized_colors = plt.cm.viridis((filtered_colors / np.max(filtered_colors))*2.5)[:, :3]
                else:
                    normalized_colors = filtered_colors.copy()
                    if normalized_colors.max() > 1:
                        normalized_colors = normalized_colors / 255.0
                
                ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2],
                          c=normalized_colors, s=1000*point_size, alpha=0.6)
            
            # Plot mesh if provided
            if mesh_vertices is not None and mesh_faces is not None:
                mesh = []
                for face in mesh_faces:
                    vertices_face = mesh_vertices[face]
                    mesh.append(vertices_face)
                mesh_collection = Poly3DCollection(mesh,
                                        facecolors=mesh_color,
                                        alpha=0.6,
                                        edgecolor='white',
                                        linewidth=1.,
                                        antialiased=True,
                                        shade=True)
                ax.add_collection3d(mesh_collection)
            
            # Calculate bounding box
            max_coords = np.array([0.8, 1., 0.45]) #if max_coords is None else max_coords
            min_coords = np.array([0, 0., 0]) #if min_coords is None else min_coords
            

            # if pointcloud is not None and mesh_vertices is not None:
            #     # Compute the min and max across both arrays
            #     min_coords = np.minimum(np.min(pointcloud, axis=0), np.min(mesh_vertices, axis=0)) #- 0.1
            #     max_coords = np.maximum(np.max(pointcloud, axis=0), np.max(mesh_vertices, axis=0)) #+ 0.1
            # elif pointcloud is not None:
            #     min_coords = np.min(pointcloud, axis=0) #- 0.1
            #     max_coords = np.max(pointcloud, axis=0) #+ 0.1
            # elif mesh_vertices is not None:
            #     min_coords = np.min(mesh_vertices, axis=0) #- 0.1
            #     max_coords = np.max(mesh_vertices, axis=0) #+ 0.1
            
            
            # Create bounding box vertices
            bbox = np.array([
                [min_coords[0], min_coords[1], min_coords[2]],
                [max_coords[0], min_coords[1], min_coords[2]],
                [max_coords[0], max_coords[1], min_coords[2]],
                [min_coords[0], max_coords[1], min_coords[2]],
                [min_coords[0], min_coords[1], max_coords[2]],
                [max_coords[0], min_coords[1], max_coords[2]],
                [max_coords[0], max_coords[1], max_coords[2]],
                [min_coords[0], max_coords[1], max_coords[2]]
            ])
            
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
            ]
            
            for i, edge in enumerate(edges):
                if i ==  11 or i==8 or i in (4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16):
                    ax.plot([bbox[edge[0], 0], bbox[edge[1], 0]],
                        [bbox[edge[0], 1], bbox[edge[1], 1]],
                        [bbox[edge[0], 2], bbox[edge[1], 2]],
                        color=[1, 0, 0, 0.8], linewidth=1, zorder=100000)
                    
                else:
                    ax.plot([bbox[edge[0], 0], bbox[edge[1], 0]],
                       [bbox[edge[0], 1], bbox[edge[1], 1]],
                       [bbox[edge[0], 2], bbox[edge[1], 2]],
                       color=[1, 0, 0, 0.8], linewidth=1)
            
            # Style adjustments
            ax.set_facecolor('white')
            ax.grid(False)
            ax.tick_params(axis='both', which='both',
                          left=False, right=False, bottom=False, top=False,
                          labelleft=False, labelbottom=False)
            
            # Add subtitle if provided
            if subtitle:
                ax.set_title(subtitle, fontsize=12)
            
            # Set axis properties
            ax.zaxis.set_pane_color((0.2, 0.2, 0.2, 1))
            ax.xaxis.set_pane_color((0, 0, 0, 1))
            ax.yaxis.set_pane_color((0, 0, 0, 1))
            
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(min_coords[0], max_coords[0])
            ax.set_ylim(min_coords[1], max_coords[1])
            ax.set_zlim(min_coords[2], max_coords[2])
    
    if images:
        views = [
            ('left', 20, 40),
            # ('front', 20, 90),
            # ('right', 20, 130)
        ]
        
        setup_scenes()
        plt.tight_layout()
        
        saved_images = []
        for view_name, elev, azim in views:
            # Set view for all subplots
            for ax in axes:
                ax.view_init(elev=elev, azim=azim)
            
            output_image = f"{os.path.splitext(output_file)[0]}_{view_name}.png"
            plt.savefig(output_image)
            
            if os.path.exists(output_image):
                print(f"Image created successfully: {output_image}")
                print(f"File size: {os.path.getsize(output_image)} bytes")
                saved_images.append(output_image)
            else:
                print(f"Error: Image file {output_image} was not created")
        
        plt.close()
        return saved_images
    
    else:
        n_frames = fps * duration
        base_angles = np.linspace(0, 360, n_frames, endpoint=False)
        angles = list(base_angles)

        setup_scenes()
        plt.tight_layout()

        # Create temporary directory for frames
        import tempfile
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            print("Generating frames...")
            # First generate the base frames
            base_frames = []
            for i, angle in enumerate(tqdm(angles)):
                # Update view for all subplots
                for ax in axes:
                    ax.view_init(elev=20, azim=angle)
                
                frame_path = os.path.join(tmpdir, f'frame_{i:04d}.png')
                plt.savefig(frame_path)
                base_frames.append(frame_path)
            
            # Now create the extended frames for forward-backward motion
            frame_count = len(base_frames)
            for i in range(1, forward_backward):
                if i % 2 == 0:
                    # Save forward frames
                    for j, frame in enumerate(base_frames):
                        new_frame_path = os.path.join(tmpdir, f'frame_{(frame_count + j):04d}.png')
                        os.system(f'cp "{frame}" "{new_frame_path}"')
                        frame_count += 1
                else:
                    # Save reversed frames
                    for j, frame in enumerate(reversed(base_frames)):
                        new_frame_path = os.path.join(tmpdir, f'frame_{(frame_count + j):04d}.png')
                        os.system(f'cp "{frame}" "{new_frame_path}"')
                        frame_count += 1
            
            print(f"Saving video to {output_file}...")
            # Use ffmpeg with the total number of frames
            cmd = ['ffmpeg', '-y', '-framerate', str(fps), 
                  '-i', os.path.join(tmpdir, 'frame_%04d.png'),
                  '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                  '-b:v', '2000k', output_file]
            subprocess.run(cmd)

        plt.close()

        if os.path.exists(output_file):
            print(f"Video created successfully: {output_file}")
            print(f"File size: {os.path.getsize(output_file)} bytes")
        else:
            print("Error: Video file was not created")
            
        return output_file


def print_images(images, n_cols, save_path):
    import matplotlib.pyplot as plt
    import math
    
    n_images = len(images)
    n_rows = math.ceil(n_images / n_cols)
    
    plt.figure(figsize=(n_cols * 4, n_rows * 4))
    
    for i, img in enumerate(images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
    
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()