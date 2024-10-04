import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.base_model import Model
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.pipelines.base_pipeline import Pipeline
import matplotlib.pyplot as plt
import os
from pathlib import Path

"""
TODO: modify config_path and save_dir
"""

# Setup the entire pipeline and load the NeRF model from a config file
def load_nerf_model(config_path):
    config_path = Path(config_path)
    _, pipeline, _, _ = eval_setup(config_path)
    return pipeline

# Create a set of viewpoints on a sphere that can be used as camera locations for rendering the NeRF model
def generate_sphere_points(num_points, radius=1):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2*np.pi, num_points)
    
    # Converte spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return np.column_stack((x, y, z))

# Visualize the camera positions by creating a 3D scatter plot showing where the camera positions are located in space
def plot_sphere_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"Camera Positions in 3D Space")
    plt.show()    

# Create a camera-to-world matrix
# 4x4 transformation matrix that represents the camera's position and orientation in world space
# position: 3D coordinates of the camera in world space
# look_at: The point the camera is looking at
def create_camera_to_world(position, look_at=[0, 0, 0], up=[0, 1, 0]):
    forward = np.array(look_at) - np.array(position)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    cam_to_world = np.eye(4)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = -forward
    cam_to_world[:3, 3] = position
    return cam_to_world

# Render an image using the NeRF model
def render_image(pipeline, camera_to_world, height, width):

    # camera_to_world_tensor = torch.tensor(camera_to_world).unsqueeze(0)  # Shape will be (1, 4, 4)
    # print("camera_to_world shape:", camera_to_world_tensor.shape)  # Should now print (1, 4, 4)


    camera = Cameras(
        camera_to_worlds=torch.tensor(camera_to_world).unsqueeze(0).float(),
        fx=float(width),  # Focal lengths in pixels
        fy=float(width),  # Focal lengths in pixels
        cx=float(width) / 2,  # Principal point (usually the image center)
        cy=float(height) / 2,  # Principal point (usually the image center)
        height=int(height),  # Convert height to int
        width=int(width),    # Convert width to int
    )
    
    # Ray generation (based on the camera parameters provided)
    # Produce the rendered output (this is where the actual rendering happens)
    outputs = pipeline.model.get_outputs_for_camera(camera)

    # Extract the color information from rendered results
    rgb = outputs["rgb"].reshape(height, width, 3)
    # return outputs
    return rgb.cpu().numpy()

def main():
    config_path = "/home/tchen604/nerfstudio/outputs/poster/nerfacto/2024-09-28_153540/config.yml"
    pipeline = load_nerf_model(config_path)
    
    num_cameras = 50
    radius = 2
    height, width = 800, 800  # Output image dimensions
    # Convert to float
    # height, width = float(height), float(width)
    
    # Generate and plot camera positions
    camera_positions = generate_sphere_points(num_cameras, radius)
    plot_sphere_points(camera_positions)

    save_dir = "/home/tchen604/nerfstudio/outputs/poster/nerfacto/2024-09-28_153540/nerfscope_save"
    os.makedirs(save_dir, exist_ok=True)

    # Prepare a tensor to hold all camera_to_world matrices
    camera_to_worlds = []
    
    # Use the camera positions for rendering
    # for i, position in enumerate(camera_positions):
    for position in camera_positions:
        camera_to_world = create_camera_to_world(position)

        # Convert to a tensor
        if isinstance(camera_to_world, np.ndarray):
            camera_to_world = torch.tensor(camera_to_world, dtype=torch.float32)

        camera_to_worlds.append(camera_to_world)

    # Convert to a tensor and add batch dimension
    camera_to_worlds = torch.stack(camera_to_worlds)  # Shape should now be (num_cameras, 4, 4)


        # Ensure it's a tensor
        # if not isinstance(camera_to_world, torch.Tensor):
        #     camera_to_world = torch.tensor(camera_to_world, dtype=torch.float32)
    num_rays = 4096  # Set this to your actual number of rays
    for i in range(num_cameras):
        
        # camera_to_world_batch = camera_to_worlds[i].unsqueeze(0).expand(num_rays, -1, -1)  # (num_rays, 4, 4)
        # rgb = render_image(pipeline, camera_to_world_batch, height, width)
        rgb = render_image(pipeline, camera_to_worlds, height, width)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb)
        plt.axis('off')
        plt.title(f"Render from camera position {i+1}")
        plt.savefig(os.path.join(save_dir, f"render_{i+1}.png"))
        plt.close()
        
        # Provide progress feedback
        CONSOLE.print(f"Rendered image {i+1}/{num_cameras}")

if __name__ == "__main__":
    main()
    