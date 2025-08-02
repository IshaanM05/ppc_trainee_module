import numpy as np
import matplotlib.pyplot as plt

# Given data points
x = [1, 2, 3, 4]
y = [1, 3, 2, 4]

x1, x2, x3, x4 = x
y1, y2, y3, y4 = y

# Build the coefficient matrix A manually (size 12x12 for 3 cubic splines)
A = np.array([
    # Spline 1 (from x1 to x2)
    [1, x1, x1**2, x1**3, 0, 0, 0, 0, 0, 0, 0, 0],  # S1(x1) = y1
    [1, x2, x2**2, x2**3, 0, 0, 0, 0, 0, 0, 0, 0],  # S1(x2) = y2

    # Spline 2 (from x2 to x3)
    [0, 0, 0, 0, 1, x2, x2**2, x2**3, 0, 0, 0, 0],  # S2(x2) = y2
    [0, 0, 0, 0, 1, x3, x3**2, x3**3, 0, 0, 0, 0],  # S2(x3) = y3

    # Spline 3 (from x3 to x4)
    [0, 0, 0, 0, 0, 0, 0, 0, 1, x3, x3**2, x3**3],  # S3(x3) = y3
    [0, 0, 0, 0, 0, 0, 0, 0, 1, x4, x4**2, x4**3],  # S3(x4) = y4

    # First derivative continuity at x2: S1'(x2) = S2'(x2)
    [0, 1, 2*x2, 3*x2**2, -0, -1, -2*x2, -3*x2**2, 0, 0, 0, 0],

    # First derivative continuity at x3: S2'(x3) = S3'(x3)
    [0, 0, 0, 0, 0, 1, 2*x3, 3*x3**2, -0, -1, -2*x3, -3*x3**2],

    # Second derivative continuity at x2: S1''(x2) = S2''(x2)
    [0, 0, 2, 6*x2, 0, 0, -2, -6*x2, 0, 0, 0, 0],

    # Second derivative continuity at x3: S2''(x3) = S3''(x3)
    [0, 0, 0, 0, 0, 0, 2, 6*x3, 0, 0, -2, -6*x3],

    # Natural spline boundary condition: S1''(x1) = 0
    [0, 0, 2, 6*x1, 0, 0, 0, 0, 0, 0, 0, 0],

    # Natural spline boundary condition: S3''(x4) = 0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6*x4]
])

# RHS vector B (match y values and zero second derivatives)
B = np.array([y1, y2, y2, y3, y3, y4, 0, 0, 0, 0, 0, 0])

# Solve the system Ax = B for spline coefficients
coeffs = np.linalg.solve(A, B)

# Extract coefficients for the three splines
c1 = coeffs[0:4]
c2 = coeffs[4:8]
c3 = coeffs[8:12]

# Define a function to evaluate cubic spline
def eval_spline(c, x_val):
    return c[0] + c[1]*x_val + c[2]*x_val**2 + c[3]*x_val**3

# Plotting
x_vals = np.linspace(x1, x2, 100)
y_vals = eval_spline(c1, x_vals)

x_vals2 = np.linspace(x2, x3, 100)
y_vals2 = eval_spline(c2, x_vals2)

x_vals3 = np.linspace(x3, x4, 100)
y_vals3 = eval_spline(c3, x_vals3)

plt.plot(x_vals, y_vals,color='black')
plt.plot(x_vals2, y_vals2,color='black')
plt.plot(x_vals3, y_vals3,color='black')
plt.scatter(x, y, color='red', label='Data points')
plt.title('Cubic Spline Interpolation (Manual)')
plt.legend()
plt.grid(True)
plt.show()



