import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from itertools import product
import random
import pandas as pd
import seaborn as sns


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

def perform_visibility_analysis(data_cube, acc_cube, raycast_size):
    """
    Perform visibility analysis with data cube placed in middle-bottom of accumulator
    Increases accumulator values outside and above the data cube
    """
    data_size = data_cube.shape[0]
    acc_size = acc_cube.shape[0]

    # Calculate exact middle position
    start_xy = (acc_size - data_size) // 2
    end_xy = start_xy + data_size

    # For each non-zero voxel in data cube
    for x, y, z in product(range(data_size), repeat=3):
        if data_cube[x,y,z] != 0:
            # Place in middle of accumulator cube
            data_point = (x + start_xy, y + start_xy, z)

            # For each point in accumulator cube that's outside data cube
            for acc_x in range(acc_size):
                for acc_y in range(acc_size):
                    for acc_z in range(data_size, acc_size):  # Start from top of data cube
                        ray_point = (acc_x, acc_y, acc_z)
                        
                        # Skip if point is inside data cube region
                        if (start_xy <= acc_x < end_xy and 
                            start_xy <= acc_y < end_xy and 
                            acc_z < data_size):
                            continue

                        line_points = bresenham_3d(data_point, ray_point)
                        blocked = False

                        # Check if line passes through any data points
                        for px, py, pz in line_points[1:]:
                            check_x = px - start_xy
                            check_y = py - start_xy
                            check_z = pz

                            if (0 <= check_x < data_size and
                                0 <= check_y < data_size and
                                0 <= check_z < data_size):
                                if data_cube[check_x, check_y, check_z] != 0:
                                    blocked = True
                                    break

                        if not blocked:
                            # Double the value for points above data cube
                            multiplier = 2 if ray_point[2] >= data_size else 1
                            acc_cube[ray_point] += multiplier

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
    Plot accumulator values for each XY surface
    """
    z_levels = min(6, acc_cube.shape[2])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i in range(z_levels):
        sns.heatmap(acc_cube[:,:,i], ax=axes[i], cmap='viridis', 
                   vmin=0, vmax=np.max(acc_cube))  # Use same scale for all plots
        axes[i].set_title(f'Visibility at Z = {i}')

    plt.tight_layout()
    plt.show()

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

def main():
    # Input parameters
    data_size = 5  # 5x5x5 data cube
    acc_size = 20  # 20x20x20 accumulator
    raycast_size = 3
    probability = 0.05  

    # Create cubes
    data_cube = create_data_cube(data_size, probability)
    acc_cube = np.zeros((acc_size, acc_size, acc_size))

    # Perform visibility analysis
    acc_cube = perform_visibility_analysis(data_cube, acc_cube, raycast_size)

    # Print maximum value and its position
    max_val = np.max(acc_cube)
    max_pos = np.unravel_index(np.argmax(acc_cube), acc_cube.shape)
    print(f"\nMaximum visibility value: {max_val}")
    print(f"Position of maximum value: {max_pos}")

    # Print accumulator values
    print_accumulator_values(acc_cube)

    # Plot accumulator surfaces
    plot_accumulator_surfaces(acc_cube)

    # Find top 5 visibility points
    flat_indices = np.argsort(acc_cube.ravel())[-5:]
    top_positions = np.unravel_index(flat_indices, acc_cube.shape)
    top_positions = list(zip(*top_positions))

    # Visualize results
    visualize_results(data_cube, acc_cube, top_positions)

if __name__ == "__main__":
    main()