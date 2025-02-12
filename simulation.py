import numpy as np
import plotly.graph_objects as go

def run_simulation(vertical_res=100, horizontal_res=100, probability=0.5):
    """
    Runs a simple simulation and returns two plotly figures.
    
    Parameters:
    -----------
    vertical_res : int
        Vertical resolution of the simulation grid
    horizontal_res : int
        Horizontal resolution of the simulation grid
    probability : float
        Probability parameter for the simulation
    
    Returns:
    --------
    tuple
        (fig1, fig2) - Two plotly figure dictionaries
    """
    # Generate some sample data
    x = np.linspace(0, 10, horizontal_res)
    y = np.linspace(0, 10, vertical_res)
    X, Y = np.meshgrid(x, y)
    
    # Create first visualization (heatmap)
    Z1 = np.sin(X) * np.cos(Y) * np.random.random() * probability
    fig1 = go.Figure(data=go.Heatmap(z=Z1))
    fig1.update_layout(
        title='Simulation Heatmap',
        xaxis_title='X Axis',
        yaxis_title='Y Axis'
    )
    
    # Create second visualization (surface)
    Z2 = np.cos(X) * np.sin(Y) * np.random.random() * probability
    fig2 = go.Figure(data=go.Surface(z=Z2))
    fig2.update_layout(
        title='Simulation Surface',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )
    
    # Convert figures to dictionaries for JSON serialization
    return fig1.to_dict(), fig2.to_dict()

if __name__ == "__main__":
    # Test the simulation
    fig1, fig2 = run_simulation()
    print("Simulation test successful") 