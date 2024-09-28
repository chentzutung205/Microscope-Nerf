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

"""
TODO: modify config_path and save_dir
"""

# Setup the entire pipeline and load the NeRF model from a config file
def load_nerf_model(config_path):
    pipeline, _, _ = eval_setup(config_path)
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
    camera = Cameras(
        camera_to_worlds=torch.tensor(camera_to_world).unsqueeze(0).float(),
        fx=width,  # Focal lengths in pixels
        fy=width,  # Focal lengths in pixels
        cx=width/2,  # Principal point (usually the image center)
        cy=height/2,  # Principal point (usually the image center)
        height=height,
        width=width,
    )
    
    # Ray generation (based on the camera parameters provided)
    # Produce the rendered output (this is where the actual rendering happens)
    outputs = pipeline.model.get_outputs_for_camera(camera)

    # Extract the color information from rendered results
    rgb = outputs["rgb"].reshape(height, width, 3)

    return rgb.cpu().numpy()

def main():
    config_path = "path/to/your/config.yml"
    pipeline = load_nerf_model(config_path)
    
    num_cameras = 50
    radius = 2
    height, width = 800, 800  # Output image dimensions
    
    # Generate and plot camera positions
    camera_positions = generate_sphere_points(num_cameras, radius)
    plot_sphere_points(camera_positions)

    save_dir = "path/to/your/desired/folder"
    os.makedirs(save_dir, exist_ok=True)
    
    # Use the camera positions for rendering
    for i, position in enumerate(camera_positions):
        camera_to_world = create_camera_to_world(position)
        rgb = render_image(pipeline, camera_to_world, height, width)
        
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
    