import sys
from visibility_analysis import create_data_cube, perform_visibility_analysis, export_to_threejs
import numpy as np

def run_with_params(data_size, acc_size, raycast_size, probability):
    # Create cubes
    data_cube = create_data_cube(data_size, probability)
    acc_cube = np.zeros((acc_size, acc_size, acc_size))

    # Perform visibility analysis
    acc_cube = perform_visibility_analysis(data_cube, acc_cube, raycast_size)

    # Find top 5 visibility points
    flat_indices = np.argsort(acc_cube.ravel())[-5:]
    top_positions = np.unravel_index(flat_indices, acc_cube.shape)
    top_positions = list(zip(*top_positions))

    # Export data for Three.js
    export_to_threejs(data_cube, acc_cube, top_positions)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run_analysis.py data_size acc_size raycast_size probability")
        sys.exit(1)
    
    data_size = int(sys.argv[1])
    acc_size = int(sys.argv[2])
    raycast_size = int(sys.argv[3])
    probability = float(sys.argv[4])
    
    run_with_params(data_size, acc_size, raycast_size, probability) 