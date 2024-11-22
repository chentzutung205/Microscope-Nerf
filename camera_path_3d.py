import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_upper_hemisphere_path(radius=1.0, num_points=100):
    """
    Generate points along the upper hemisphere of a sphere.

    Parameters:
        radius (float): Radius of the sphere.
        num_points (int): Number of points along the path.

    Returns:
        np.ndarray: Array of 3D points on the upper hemisphere.
    """

    # Generate points evenly distributed along the upper hemisphere
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)  # Elevation angles
    phi = 0    # Azimuthal angles

    # Create a grid for theta and phi
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Convert spherical to Cartesian coordinates
    x = radius * np.sin(theta_grid) * np.cos(phi_grid)
    y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = radius * np.cos(theta_grid)

    # Flatten and combine into a list of points
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    return points

def visualize_sphere_with_path(radius=1.0, path_points=None):
    """
    Visualize the sphere, path, and points along the upper hemisphere.

    Parameters:
        radius (float): Radius of the sphere.
        path_points (np.ndarray): Array of 3D points along the path.
    """

    # Create the sphere
    phi = np.linspace(0, 2 * np.pi, 30)
    theta = np.linspace(0, np.pi, 30)
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Plot the sphere
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='white', alpha=0.3, edgecolor='gray')

    # Plot the path
    if path_points is not None:
        path_x, path_y, path_z = path_points.T
        ax.scatter(path_x, path_y, path_z, color='red', s=20, label="Path Points")
        ax.plot(path_x, path_y, path_z, color='blue', label="Path Trajectory")

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Upper Hemisphere Path")
    ax.legend()
    plt.show()

# Generate path points
radius = 2.0
num_points = 15
path_points = generate_upper_hemisphere_path(radius, num_points)

# Visualize the sphere and path
visualize_sphere_with_path(radius, path_points)
