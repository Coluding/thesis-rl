import numpy as np
import plotly.graph_objects as go

# Generate data
x = np.cos(np.linspace(0, 2*np.pi,  100))
y = np.sin(np.linspace(0, 2*np.pi, 100))

x, y = np.meshgrid(x, y)
z = 3 * x**2 * y - x*y + y**3

# Create the surface plot
surface = go.Surface(
    z=z, x=x, y=y, colorscale='Viridis', showscale=False,
    contours={
        "z": {
            "show": False,  # Disable background contours
            "usecolormap": True,
            "highlightcolor": "limegreen",
            "project": {"z": True}
        }
    }
)

# Generate wireframe lines
wireframe_lines = []
for i in range(len(x)):
    wireframe_lines.append(go.Scatter3d(x=x[i, :], y=y[i, :], z=z[i, :], mode='lines', line=dict(color='black', width=1)))
    wireframe_lines.append(go.Scatter3d(x=x[:, i], y=y[:, i], z=z[:, i], mode='lines', line=dict(color='black', width=1)))

# Create contour plot on the bottom
contours = go.Contour(z=z, x=x[0], y=y[:, 0], colorscale='Viridis', showscale=False, line=dict(width=2))

# Combine all traces
data = [surface] + wireframe_lines + [contours]

# Define layout
layout = go.Layout(
    title='3D Surface Plot with Wireframe and Contours',
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        camera=dict(
            eye=dict(x=1.87, y=0.88, z=-0.64)
        )
    ),
    autosize=True,  # Make the graph full screen
    margin=dict(l=0, r=0, b=0, t=30)  # Adjust margins for full screen
)

# Create the figure and display it
fig = go.Figure(data=data, layout=layout)
fig.show()
