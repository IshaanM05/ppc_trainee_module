import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# 1. Define waypoints
waypoints = np.array([
    [0, 0],
    [1, 2],
    [3, 3],
    [4, 0]
])
x = waypoints[:, 0]
y = waypoints[:, 1]

# 2. Arc-length parameterization for t
t = np.zeros(len(x))
for i in range(1, len(x)):
    t[i] = t[i-1] + np.hypot(x[i] - x[i-1], y[i] - y[i-1])

# 3. Compute curvature
def compute_curvature(x_t, y_t, t_vals):
    dx = np.gradient(x_t, t_vals)
    dy = np.gradient(y_t, t_vals)
    ddx = np.gradient(dx, t_vals)
    ddy = np.gradient(dy, t_vals)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**1.5 + 1e-8  # Avoid div by 0
    curvature = numerator / denominator
    return curvature

# 4. Cost function (curvature squared)
def cost_function(b0_x, bn_x, b0_y, bn_y, t, x, y, t_fine):
    cs_x = CubicSpline(t, x, bc_type=((2, b0_x), (2, bn_x)))
    cs_y = CubicSpline(t, y, bc_type=((2, b0_y), (2, bn_y)))

    x_vals = cs_x(t_fine)
    y_vals = cs_y(t_fine)

    curvature = compute_curvature(x_vals, y_vals, t_fine)
    cost = np.trapz(curvature**2, t_fine)  # Integral of curvature squared
    return cost

# 5. Optimization function using scipy.optimize
def optimize_spline(t, x, y, t_fine):
    # Initial guess for second derivatives at the boundaries
    initial_guess = [0.0, 0.0, 0.0, 0.0]
    
    # Optimize the second derivatives at the boundaries using minimize
    result = minimize(
        lambda params: cost_function(params[0], params[1], params[2], params[3], t, x, y, t_fine),
        initial_guess, 
        method='BFGS'
    )

    # Return the optimized boundary conditions
    return result.x

# 6. Run optimization
t_fine = np.linspace(t[0], t[-1], 200)
optimized_boundary_conditions = optimize_spline(t, x, y, t_fine)

# Extract the optimized second derivatives at the boundaries
b0_x, bn_x, b0_y, bn_y = optimized_boundary_conditions

# 7. Generate optimized spline
cs_x_opt = CubicSpline(t, x, bc_type=((2, b0_x), (2, bn_x)))
cs_y_opt = CubicSpline(t, y, bc_type=((2, b0_y), (2, bn_y)))
x_opt = cs_x_opt(t_fine)
y_opt = cs_y_opt(t_fine)

# 8. Generate original spline (for comparison)
cs_x_nat = CubicSpline(t, x, bc_type='natural')
cs_y_nat = CubicSpline(t, y, bc_type='natural')
x_nat = cs_x_nat(t_fine)
y_nat = cs_y_nat(t_fine)

# 9. Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro', label='Waypoints')
plt.plot(x_nat, y_nat, 'g--', label='Original Spline (Natural)')
plt.plot(x_opt, y_opt, 'b-', label='Optimized Spline (Min Curvature)')
plt.title("Optimized vs Original Cubic Spline")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

