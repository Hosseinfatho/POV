from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
from itertools import product
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dask.array as da
import zarr
import requests
from ome_zarr.io import parse_url
import ome_types
from vitessce import VitessceConfig, CoordinationLevel as CL, get_initial_coordination_scope_prefix
from real_data import load_and_analyze_data

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

def create_data_cube(size, probability):
    colors = ['green', 'blue', 'red', 'yellow']
    cube = np.zeros((size, size, size), dtype=object)
    for x, y, z in product(range(size), repeat=3):
        if random.random() < probability/100:
            cube[x,y,z] = random.choice(colors)
    return cube

def bresenham_3d(start, end):
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
    directions = []
    phi_step = 360 / vertical_res
    theta_step = 180 / horizontal_res

    for phi in np.arange(0, 360, phi_step):
        for theta in np.arange(0, 180, theta_step):
            phi_rad = np.radians(phi)
            theta_rad = np.radians(theta)
            x = np.sin(theta_rad) * np.cos(phi_rad)
            y = np.sin(theta_rad) * np.sin(phi_rad)
            z = np.cos(theta_rad)
            directions.append((x, y, z))

    return directions

def perform_visibility_analysis(data_cube, acc_cube, raycast_cube_size):
    data_size = data_cube.shape[0]
    acc_size = acc_cube.shape[0]
    start_xy = (acc_size - data_size) // 2
    
    # Ensure raycast_cube_size is odd
    if raycast_cube_size % 2 == 0:
        raycast_cube_size += 1
    
    # Calculate center offset for raycast cube
    rc_offset = raycast_cube_size // 2
    
    # For each data point
    for x, y, z in product(range(data_size), repeat=3):
        if data_cube[x,y,z] != 0:  # If this is a data point
            center_x = x + start_xy
            center_y = y + start_xy
            center_z = z
            
            # Create a set to track unique voxels hit by this data point
            hit_voxels = set()
            
            # Get top face centers of raycast cube
            top_z = z + rc_offset
            for rx in range(-rc_offset, rc_offset + 1):
                for ry in range(-rc_offset, rc_offset + 1):
                    target_x = center_x + rx
                    target_y = center_y + ry
                    
                    # Calculate direction vector
                    dx = target_x - center_x
                    dy = target_y - center_y
                    dz = rc_offset  # Distance to top face
                    
                    # Normalize direction vector
                    length = np.sqrt(dx*dx + dy*dy + dz*dz)
                    if length > 0:
                        dx /= length
                        dy /= length
                        dz /= length
                    
                    # Cast ray from data point through raycast cube top face
                    max_distance = acc_size * 2  # Ensure ray reaches acc cube boundaries
                    for dist in range(1, max_distance):
                        ray_x = int(center_x + dx * dist)
                        ray_y = int(center_y + dy * dist)
                        ray_z = int(center_z + dz * dist)
                        
                        # Check if ray is within acc_cube bounds
                        if (0 <= ray_x < acc_size and 
                            0 <= ray_y < acc_size and 
                            ray_z > data_size + raycast_cube_size):  # Only count points above data + raycast cube
                            
                            # Check if ray hits another data point
                            check_x = ray_x - start_xy
                            check_y = ray_y - start_xy
                            check_z = ray_z - start_xy
                            
                            blocked = False
                            if (0 <= check_x < data_size and 
                                0 <= check_y < data_size and 
                                0 <= check_z < data_size):
                                if (data_cube[check_x, check_y, check_z] != 0 and 
                                    (check_x != x or check_y != y or check_z != z)):
                                    blocked = True
                                    break
                            
                            if not blocked and ray_z < acc_size:
                                voxel = (ray_x, ray_y, ray_z)
                                if voxel not in hit_voxels:
                                    hit_voxels.add(voxel)
                                    acc_cube[ray_x, ray_y, ray_z] += 1
                        
                        # Stop if ray exits acc_cube
                        if (ray_x < 0 or ray_x >= acc_size or
                            ray_y < 0 or ray_y >= acc_size or
                            ray_z >= acc_size):
                            break
    
    return acc_cube

def create_visualization_data(params):
    try:
        # Get parameters
        data_size = int(params['dataSize'])
        acc_size = int(params['accSize'])
        raycast_cube_size = int(params.get('raycastCubeSize', 5))
        probability = float(params['probability'])

        # Create data cube and accumulator
        data_cube = create_data_cube(data_size, probability)
        acc_cube = np.zeros((acc_size, acc_size, acc_size))
        acc_cube = perform_visibility_analysis(data_cube, acc_cube, raycast_cube_size)

        # Create 3D scatter plot
        start_xy = (acc_size - data_size) // 2
        x_data, y_data, z_data, colors = [], [], [], []
        color_counts = {'green': 0, 'blue': 0, 'red': 0, 'yellow': 0}
        for x, y, z in product(range(data_size), repeat=3):
            if data_cube[x,y,z] != 0:
                x_data.append(x + start_xy)
                y_data.append(y + start_xy)
                z_data.append(z)
                colors.append(data_cube[x,y,z])
                color_counts[data_cube[x,y,z]] += 1

        # Create first figure (3D scatter)
        fig1_data = [{
            'type': 'scatter3d',
            'x': x_data,
            'y': y_data,
            'z': z_data,
            'mode': 'markers',
            'marker': {
                'size': 8,
                'color': colors,
                'opacity': 0.8
            },
            'name': 'Data Points',
            'showlegend': True,
            'legendgroup': 'data'
        }]

        # Add camera positions
        flat_indices = np.argsort(acc_cube.ravel())[-5:]
        top_positions = np.unravel_index(flat_indices, acc_cube.shape)
        top_positions = list(zip(*top_positions))

        for i, (x, y, z) in enumerate(top_positions):
            fig1_data.append({
                'type': 'scatter3d',
                'x': [float(x)],
                'y': [float(y)],
                'z': [float(z)],
                'mode': 'markers',
                'marker': {
                    'size': 12,
                    'color': 'black',
                    'symbol': 'circle'
                },
                'name': f'Camera {i+1} (Value: {float(acc_cube[x,y,z]):.0f})',
                'showlegend': True,
                'legendgroup': f'camera{i+1}'
            })

        # Create second figure (heatmap)
        fig2_data = [{
            'type': 'heatmap',
            'z': acc_cube[:,:,0].tolist(),
            'colorscale': 'Viridis',
            'colorbar': {
                'title': 'Visibility Count',
                'tickmode': 'array',
                'ticktext': [str(i) for i in range(int(np.max(acc_cube))+1)],
                'tickvals': [i for i in range(int(np.max(acc_cube))+1)],
            }
        }]
        # Create slider steps
        sliders = [{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Z-Level: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [{
                'args': [
                    {'z': [acc_cube[:,:,i].tolist()]},
                    {
                        'title': f'Accumulator Values at Z={i}',
                        'colorbar.tickmode': 'array',
                        'colorbar.ticktext': [str(i) for i in range(int(np.max(acc_cube))+1)],
                        'colorbar.tickvals': [i for i in range(int(np.max(acc_cube))+1)]
                    }
                ],
                'label': str(i),
                'method': 'update'
            } for i in range(acc_cube.shape[2])]
        }]

        # Return the complete visualization data
        return {
            'fig1': {
                'data': fig1_data,
                'layout': {
                    'scene': {
                        'aspectmode': 'cube',
                        'xaxis': {'range': [0, acc_size]},
                        'yaxis': {'range': [0, acc_size]},
                        'zaxis': {'range': [0, acc_size]},
                        'camera': {
                            'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}  # Adjust camera view
                        }
                    },
                    'width': 800,
                    'height': 800,
                    'title': 'Data Points and Camera Positions',
                    'showlegend': True,
                    'legend': {
                        'x': 1.1,
                        'y': 1.0,
                        'bgcolor': 'rgba(255, 255, 255, 0.8)',
                        'bordercolor': 'rgba(0, 0, 0, 0.2)',
                        'borderwidth': 1
                    }
                }
            },
            'fig2': {
                'data': fig2_data,
                'layout': {
                    'width': 800,
                    'height': 800,
                    'title': f'Accumulator Values (Max: {int(np.max(acc_cube))})',
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'},
                    'sliders': sliders,
                    'updatemenus': [{
                        'type': 'buttons',
                        'showactive': False,
                        'y': 0,
                        'x': 1.2,
                        'xanchor': 'right',
                        'yanchor': 'top',
                        'buttons': [{
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 500, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                            }]
                        }]
                    }]
                }
            }
        }
        # Print statistics
        print(f"Number of data points: {len(x_data)}")
        print(f"Number of cameras: {len(top_positions)}")
        print(f"Accumulator shape: {acc_cube.shape}")

        return {
            'fig1': {
                'data': fig1_data,
                'layout': {
                    'scene': {
                        'aspectmode': 'cube',
                        'xaxis': {'range': [0, acc_size]},
                        'yaxis': {'range': [0, acc_size]},
                        'zaxis': {'range': [0, acc_size]}
                    },
                    'width': 800,
                    'height': 800,
                    'title': 'Data Points and Camera Positions',
                    'showlegend': True,
                    'legend': {
                        'x': 1.1,
                        'y': 1.0,
                        'bgcolor': 'rgba(255, 255, 255, 0.8)',
                        'bordercolor': 'rgba(0, 0, 0, 0.2)',
                        'borderwidth': 1
                    }
                }
            },
            'fig2': {
                'data': fig2_data,
                'layout': {
                    'width': 800,
                    'height': 800,
                    'title': f'Accumulator Values (Max: {int(np.max(acc_cube))})',
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'}
                }
            }
        }

    except Exception as e:
        print(f"Error in create_visualization_data: {str(e)}")
        return {'error': str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    return jsonify({"status": "success", "message": "Analysis completed successfully!"})

@app.route('/get_visualization', methods=['POST'])
def get_visualization():
    try:
        params = request.json
        data_size = int(params['dataSize'])
        acc_size = int(params['accSize'])
        raycast_cube_size = int(params.get('raycastCubeSize', 5))
        probability = float(params['probability'])

        data_cube = create_data_cube(data_size, probability)
        acc_cube = np.zeros((acc_size, acc_size, acc_size))
        acc_cube = perform_visibility_analysis(data_cube, acc_cube, raycast_cube_size)

        result = create_visualization_data(params)
        return jsonify(result)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_real_data')
def get_real_data():
    try:
        vc = load_and_analyze_data()
        return jsonify({
            "config": vc.to_dict()
        })
    except Exception as e:
        print("Error in get_real_data:", str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)