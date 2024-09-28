# NerfScope
 
**NeRF Model Visualizer**

This script allows you to visualize a trained NeRF model from multiple viewpoints. It generates camera positions on a sphere around the center of the scene and renders images from these positions using the NeRF model.

Requirements:
- Python 3.8 or higher
- PyTorch
- Nerfstudio
- NumPy
- Matplotlib

Installation:
1. Ensure you have Python 3.8+ installed.
2. Install the required packages:
   pip install torch nerfstudio numpy matplotlib

Usage:
1. Place this script in a directory where you have access to your trained NeRF model.
  
3. Modify the following variables in the script as needed:
   - config_path: Path to your Nerfstudio config file
   - num_cameras: Number of viewpoints to render
   - radius: Radius of the sphere for camera positions
   - height, width: Dimensions of the rendered images
   - save_dir: Directory to save the rendered images
  
4. Run the script:
   python nerf_visualizer.py

5. The script will generate and save rendered images from different viewpoints in the specified save directory.

Key Functions:
- load_nerf_model: Loads the NeRF model from a config file
- generate_sphere_points: Generates camera positions on a sphere
- create_camera_to_world: Creates a camera-to-world transformation matrix
- render_image: Renders an image using the NeRF model for a given camera position

Output:
- Rendered images saved as PNG files in the specified directory
- Console output showing rendering progress

Notes:
- Ensure you have sufficient GPU memory to run the NeRF model.
- Rendering multiple high-resolution images may take some time.
- Adjust the radius parameter to change the distance of camera positions from the center of the scene.

For more information on NeRF and Nerfstudio, visit:
https://github.com/nerfstudio-project/nerfstudio
