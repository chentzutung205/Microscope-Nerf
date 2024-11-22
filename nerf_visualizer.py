import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.models.base_model import Model
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.pipelines.base_pipeline import Pipeline
import matplotlib.pyplot as plt
import os
from pathlib import Path


"""
Change if needed:
config_path
save_dir
num_cameras
radius
height, weight (image dimensions)
intrinsic parameters of camera
"""


# Setup the entire rendering pipeline and load the NeRF model from a config file
def load_nerf_model(config_path):
    config_path = Path(config_path)
    _, pipeline, _, _ = eval_setup(config_path)
    return pipeline


# Create a set of viewpoints on a sphere that can be used as camera positions for rendering the NeRF model
# Generate 'num_points' uniformly distributed points on the surface of a sphere with a specified 'radius'
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


# Construct a camera-to-world (c2w) matrix
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
    return cam_to_world[:3, :]


# Render an image using the NeRF model
def render_image(pipeline, camera_to_world, intrinsics, height, width):
    camera = Cameras(
        camera_to_worlds = camera_to_world.float(),
        # camera_to_worlds=torch.tensor(camera_to_world).unsqueeze(0).float(),
        fx=intrinsics['fx'],  # Focal lengths in pixels
        fy=intrinsics['fy'],  # Focal lengths in pixels
        cx=intrinsics['cx'],  # Principal point (usually the image center)
        cy=intrinsics['cy'],  # Principal point (usually the image center)
        height=torch.tensor([height], dtype=torch.int64),
        width=torch.tensor([width], dtype=torch.int64),
        camera_type=CameraType.PERSPECTIVE.value  # Change if using different camera types
    )
    
    # Ray generation (based on the camera parameters provided)
    # Produce the rendered output (this is where the actual rendering happens)
    outputs = pipeline.model.get_outputs_for_camera(camera)

    # Extract the color information from rendered results
    rgb = outputs["rgb"].reshape(height, width, 3)

    return rgb.cpu().numpy()


def main():

    # Modify if needed
    config_path = "/home/tchen604/nerfstudio/outputs/poster/nerfacto/2024-09-28_153540/config.yml"
    save_dir = "/home/tchen604/nerfstudio/outputs/poster/nerfacto/2024-09-28_153540/nerfscope_save"
    num_cameras = 50
    radius = 1
    height, width = 1080, 1920  # Output image dimensions
    
    # Define intrinsic parameters
    intrinsic_params = {
        'fx': torch.tensor([480.6130]),
        'fy': torch.tensor([481.5445]),
        'cx': torch.tensor([324.1875]),
        'cy': torch.tensor([210.0625]),
    }
    
    # Load the NeRF model
    pipeline = load_nerf_model(config_path)
    
    # Generate and plot camera positions
    camera_positions = generate_sphere_points(num_cameras, radius)
    plot_sphere_points(camera_positions)
    
    os.makedirs(save_dir, exist_ok=True)

    # Use the camera positions for rendering
    camera_to_worlds = []
    for i, position in enumerate(camera_positions):
        camera_to_world = create_camera_to_world(position)

        # Convert to tensor if it's a numpy array
        if isinstance(camera_to_world, np.ndarray):
            camera_to_world = torch.tensor(camera_to_world, dtype=torch.float32)
        
        camera_to_worlds.append(camera_to_world)
    
    # Stack all [3, 4] matrices into a tensor of shape (num_cameras, 3, 4)
    camera_to_worlds = torch.stack(camera_to_worlds)  # Shape: (num_cameras, 3, 4)

    for i in range(num_cameras):
        camera_to_world = camera_to_worlds[i].unsqueeze(0)
        rgb = render_image(pipeline, camera_to_world, intrinsic_params, height, width)
        
        plt.figure(figsize=(20, 20))
        plt.imshow(rgb)
        plt.axis('off')
        plt.title(f"Render from camera position {i+1}")
        plt.savefig(os.path.join(save_dir, f"render_{i+1}.png"))
        plt.close()
        
        # Provide progress feedback
        CONSOLE.print(f"Rendered image {i+1}/{num_cameras}")


if __name__ == "__main__":
    main()
    