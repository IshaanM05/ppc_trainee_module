import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Step 1: Load CSV data using pandas
df = pd.read_csv('loop_track_waypoints.csv')

# Step 2: Extract the waypoints (x, y) columns
x = df['X'].to_numpy()  # X-coordinates
y = df['Y'].to_numpy()  # Y-coordinates

# Step 3: Create the parameter t, which is an increasing sequence
t = np.linspace(0, 1, len(x))

# Step 4: Perform cubic spline interpolation to generate smooth path
# Create cubic spline interpolator using t as the independent variable
cs_x = CubicSpline(t, x, bc_type='natural')
cs_y = CubicSpline(t, y, bc_type='natural')

# Step 5: Define a finer set of t-values for interpolation
# This will create a smooth curve by interpolating at smaller steps
t_new = np.linspace(0, 1, 500)

# Interpolate x and y values using the cubic spline
x_new = cs_x(t_new)
y_new = cs_y(t_new)

# Step 6: Visualize both the original waypoints and the interpolated path
plt.figure(figsize=(8, 6))

# Plot the original discrete waypoints
plt.scatter(x, y, color='red', label='Original Waypoints', zorder=5)

# Plot the interpolated smooth curve
plt.plot(x_new, y_new, color='blue', label='Interpolated Path', linewidth=2)

# Add labels and title
plt.xlabel('X-coordinate (meters)')
plt.ylabel('Y-coordinate (meters)')
plt.title('Waypoint Interpolation and Path Visualization')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()
