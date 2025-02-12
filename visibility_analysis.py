import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from itertools import product
import random
import pandas as pd
import seaborn as sns
import json


def create_data_cube(size, probability):

    colors = ['green', 'blue', 'red', 'yellow']
    cube = np.zeros((size, size, size), dtype=object)

    # Fill random voxels based on probability
    for x, y, z in product(range(size), repeat=3):
        if random.random() < probability:
            cube[x,y,z] = random.choice(colors)

    return cube

def bresenham_3d(start, end):
    """
    Implementation of 3D Bresenham's line algorithm
    """
    x1, y1, z1 = start
    x2, y2, z2 = end

    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            points.append((x1, y1, z1))
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            points.append((x1, y1, z1))
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            points.append((x1, y1, z1))
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
    points.append((x2, y2, z2))
    return points

def get_sphere_directions(vertical_res, horizontal_res):
    """
    Generate ray directions based on sphere coordinates
    vertical_res: number of vertical slices (4-36 for longitude)
    horizontal_res: number of horizontal slices (2-18 for latitude)
    """
    directions = []
    
    # Convert resolutions to angles
    phi_step = 360 / vertical_res  # Longitude (around sphere)
    theta_step = 180 / horizontal_res  # Latitude (up/down)
    
    # Generate points on sphere surface
    for phi in np.arange(0, 360, phi_step):  # Longitude
        for theta in np.arange(0, 180, theta_step):  # Latitude
            # Convert spherical coordinates to Cartesian
            phi_rad = np.radians(phi)
            theta_rad = np.radians(theta)
            
            # Calculate direction vector
            x = np.sin(theta_rad) * np.cos(phi_rad)
            y = np.sin(theta_rad) * np.sin(phi_rad)
            z = np.cos(theta_rad)
            
            directions.append((x, y, z))
    
    return directions

def is_blocked(line_points, data_cube, start_xy, data_size, source_point=None):
    """
    Check if line is blocked by any data points other than the source point
    """
    for px, py, pz in line_points[1:]:  # Skip the first point (source point)
        check_x = px - start_xy
        check_y = py - start_xy
        check_z = pz
        
        if (0 <= check_x < data_size and 
            0 <= check_y < data_size and 
            0 <= check_z < data_size):
            # Check if this is a different data point than the source
            if data_cube[check_x, check_y, check_z] != 0:
                if source_point is None or (check_x, check_y, check_z) != source_point:
                    return True
    return False
def get_ray_directions(raycast_size):
    """
    Generate ray directions from center to outer surface of raycast cube
    """
    center = raycast_size // 2
    directions = []
    
    # Generate points on all faces of the cube
    for x in range(raycast_size):
        for y in range(raycast_size):
            # Top face (z = raycast_size-1)
            directions.append((x - center, y - center, raycast_size-1 - center))
            # Bottom face (z = 0)
            if center > 0:  # Only add bottom face if not at center
                directions.append((x - center, y - center, -center))
                
    for x in range(raycast_size):
        for z in range(1, raycast_size-1):  # Skip top and bottom faces
            # Front face (y = 0)
            directions.append((x - center, -center, z - center))
            # Back face (y = raycast_size-1)
            directions.append((x - center, raycast_size-1 - center, z - center))
            
    for y in range(1, raycast_size-1):  # Skip front and back edges
        for z in range(1, raycast_size-1):  # Skip top and bottom edges
            # Left face (x = 0)
            directions.append((-center, y - center, z - center))
            # Right face (x = raycast_size-1)
            directions.append((raycast_size-1 - center, y - center, z - center))
    
    # Remove (0,0,0) if it's in the list
    directions = [(dx, dy, dz) for dx, dy, dz in directions 
                 if not (dx == 0 and dy == 0 and dz == 0)]
    
    return directions
def perform_visibility_analysis(data_cube, acc_cube, raycast_size):
    """
    Perform visibility analysis using adaptive raycast size with occlusion check:
    1. Generate rays from center to all outer voxels of raycast cube
    2. Cast rays and check for occlusion
    3. Only increment accumulator if no occlusion exists
    """
    data_size = data_cube.shape[0]
    acc_size = acc_cube.shape[0]

    # Calculate exact middle position
    start_xy = (acc_size - data_size) // 2
    
    # Get ray directions for given raycast size
    ray_directions = get_ray_directions(raycast_size)
    print(f"Number of rays: {len(ray_directions)}")
    
    # For each non-zero voxel in data cube
    for x, y, z in product(range(data_size), repeat=3):
        if data_cube[x,y,z] != 0:
            # Place data point in accumulator space
            center_x = x + start_xy
            center_y = y + start_xy
            center_z = z
            source_point = (x, y, z)  # Store original data point coordinates
            
            # Cast rays in all directions
            for dx, dy, dz in ray_directions:
                # Calculate how far to extend the ray
                max_distance = acc_size * 2  # Ensure ray reaches cube boundaries
                
                # Normalize direction vector
                length = np.sqrt(dx*dx + dy*dy + dz*dz)
                norm_dx = dx / length
                norm_dy = dy / length
                norm_dz = dz / length
                
                # Cast ray and increment points above data cube
                for dist in range(1, max_distance):
                    target_x = int(center_x + norm_dx * dist)
                    target_y = int(center_y + norm_dy * dist)
                    target_z = int(center_z + norm_dz * dist)
                    
                    # Check if point is within accumulator bounds and above data cube
                    if (0 <= target_x < acc_size and 
                        0 <= target_y < acc_size and 
                        target_z >= data_size and 
                        target_z < acc_size):
                        
                        target_point = (target_x, target_y, target_z)
                        line_points = bresenham_3d(
                            (center_x, center_y, center_z), 
                            target_point
                        )
                        
                        # Check for occlusion by other data points
                        if not is_blocked(line_points, data_cube, start_xy, data_size, source_point):
                            acc_cube[target_point] += 1
                        else:
                            # If blocked by another data point, stop this ray
                            break
                    
                    # Stop if we've gone outside the accumulator cube
                    if (target_x < 0 or target_x >= acc_size or
                        target_y < 0 or target_y >= acc_size or
                        target_z >= acc_size):
                        break

    return acc_cube

def visualize_results(data_cube, acc_cube, top_positions):
    """
    Visualize results with Plotly
    """
    fig = go.Figure()

    # Calculate offset for data cube visualization
    acc_size = acc_cube.shape[0]
    data_size = data_cube.shape[0]
    start_xy = (acc_size - data_size) // 2

    # Display data cube in its actual position in accumulator
    for x, y, z in product(range(data_size), repeat=3):
        if data_cube[x,y,z] != 0:
            fig.add_trace(go.Scatter3d(
                x=[x + start_xy],
                y=[y + start_xy],
                z=[z],
                mode='markers',
                marker=dict(
                    size=8,
                    color=data_cube[x,y,z],
                    opacity=1
                ),
                name=f'Data Point ({data_cube[x,y,z]})'
            ))

    # Display top visibility points
    for i, (x, y, z) in enumerate(top_positions):
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(
                size=12,
                color='black',
                opacity=1
            ),
            name=f'Camera {i+1} (Value: {acc_cube[x,y,z]:.0f})'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, acc_size], title='x'),
            yaxis=dict(range=[0, acc_size], title='y'),
            zaxis=dict(range=[0, acc_size], title='z'),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='white'
        ),
        showlegend=True,
        template='plotly_white'
    )

    fig.show()

def plot_accumulator_surfaces(acc_cube):
    """
    Plot accumulator values for all XY surfaces in the accumulator cube
    """
    z_levels = acc_cube.shape[2]  # Get total number of Z levels
    
    # Calculate number of rows and columns needed for subplots
    n_rows = (z_levels + 2) // 3  # Ceiling division to ensure enough rows
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.ravel()
    
    # Get global min and max for consistent colormap
    vmin = np.min(acc_cube)
    vmax = np.max(acc_cube)
    
    # Plot each Z level
    for i in range(z_levels):
        sns.heatmap(acc_cube[:,:,i], ax=axes[i], cmap='viridis',
                   vmin=vmin, vmax=vmax,
                   cbar_kws={'label': 'Visibility Count'})
        axes[i].set_title(f'Visibility at Z = {i}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
    
    # Remove empty subplots if any
    for i in range(z_levels, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Accumulator Values at Each Z Level', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics for each Z level
    print("\nSummary of values at each Z level:")
    print("Z  Min  Max  Mean  Sum")
    print("-" * 25)
    for z in range(z_levels):
        z_slice = acc_cube[:,:,z]
        print(f"{z:2d} {z_slice.min():4.0f} {z_slice.max():4.0f} {z_slice.mean():5.1f} {z_slice.sum():5.0f}")

def print_accumulator_values(acc_cube):
    """
    Print non-zero accumulator values
    """
    print("\nAccumulator Voxel Values:")
    print("X  Y  Z  Value")
    print("-" * 20)

    for x, y, z in product(range(acc_cube.shape[0]), repeat=3):
        if acc_cube[x,y,z] > 0:
            print(f"{x:2d} {y:2d} {z:2d}  {acc_cube[x,y,z]:5.0f}")

def plot_accumulator_surface_interactive(acc_cube):
    """
    Plot accumulator values for XY surfaces with an interactive Z slider
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    z_levels = acc_cube.shape[2]
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    heatmap = go.Heatmap(
        z=acc_cube[:,:,0],
        colorscale='viridis',
        zmin=np.min(acc_cube),
        zmax=np.max(acc_cube),
        colorbar=dict(title='Visibility Count')
    )
    fig.add_trace(heatmap)
    
    # Update layout with slider
    steps = []
    for i in range(z_levels):
        step = dict(
            method="update",
            args=[{"z": [acc_cube[:,:,i]]}],
            label=f"Z={i}"
        )
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Z-Level: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        title='Accumulator Values at XY Surface',
        xaxis_title='X',
        yaxis_title='Y',
        sliders=sliders,
        width=800,
        height=800
    )
    
    fig.show()
    
    # Print summary statistics for each Z level
    print("\nSummary of values at each Z level:")
    print("Z  Min  Max  Mean  Sum")
    print("-" * 25)
    for z in range(z_levels):
        z_slice = acc_cube[:,:,z]
        print(f"{z:2d} {z_slice.min():4.0f} {z_slice.max():4.0f} {z_slice.mean():5.1f} {z_slice.sum():5.0f}")

def export_to_threejs(data_cube, acc_cube, top_positions, filename='visibility_data.json'):
    """
    Export the visualization data to JSON format for Three.js
    """
    acc_size = acc_cube.shape[0]
    data_size = data_cube.shape[0]
    start_xy = (acc_size - data_size) // 2
    
    # Prepare data structure
    export_data = {
        'dataPoints': [],
        'accumulatorPoints': [],
        'topPositions': []
    }
    
    # Export data points
    for x, y, z in product(range(data_size), repeat=3):
        if data_cube[x,y,z] != 0:
            export_data['dataPoints'].append({
                'position': [x + start_xy, y + start_xy, z],
                'color': data_cube[x,y,z]
            })
    
    # Export accumulator points (only non-zero values)
    for x, y, z in product(range(acc_size), repeat=3):
        if acc_cube[x,y,z] > 0:
            export_data['accumulatorPoints'].append({
                'position': [x, y, z],
                'value': float(acc_cube[x,y,z])
            })
    
    # Export top positions
    for x, y, z in top_positions:
        export_data['topPositions'].append({
            'position': [x, y, z],
            'value': float(acc_cube[x,y,z])
        })
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(export_data, f)
    
    print(f"Data exported to {filename}")

def main():
    # Read parameters from command line or use defaults
    import sys
    import json
    
    # Default parameters
    params = {
        'dataSize': 5,
        'accSize': 20,
        'raycastSize': 3,
        'probability': 0.05
    }
    
    # Create cubes
    data_cube = create_data_cube(params['dataSize'], params['probability'])
    acc_cube = np.zeros((params['accSize'], params['accSize'], params['accSize']))

    # Perform visibility analysis
    acc_cube = perform_visibility_analysis(data_cube, acc_cube, params['raycastSize'])

    # Find top 5 visibility points
    flat_indices = np.argsort(acc_cube.ravel())[-5:]
    top_positions = np.unravel_index(flat_indices, acc_cube.shape)
    top_positions = list(zip(*top_positions))

    # Export data for Three.js
    export_to_threejs(data_cube, acc_cube, top_positions)

if __name__ == "__main__":
    main()