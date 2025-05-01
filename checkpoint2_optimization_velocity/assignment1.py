import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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
    cost = np.trapz(curvature**2, t_fine)
    return cost, x_vals, y_vals

# 5. Gradient Descent
def gradient_descent(t, x, y, t_fine, lr=0.01, iterations=100):
    b0_x, bn_x = 0.0, 0.0
    b0_y, bn_y = 0.0, 0.0
    cost_history = []

    for i in range(iterations):
        cost, _, _ = cost_function(b0_x, bn_x, b0_y, bn_y, t, x, y, t_fine)
        cost_history.append(cost)

        epsilon = 1e-5

        # Gradients (central difference)
        grad_b0x = (cost_function(b0_x + epsilon, bn_x, b0_y, bn_y, t, x, y, t_fine)[0] -
                    cost_function(b0_x - epsilon, bn_x, b0_y, bn_y, t, x, y, t_fine)[0]) / (2 * epsilon)
        grad_bnx = (cost_function(b0_x, bn_x + epsilon, b0_y, bn_y, t, x, y, t_fine)[0] -
                    cost_function(b0_x, bn_x - epsilon, b0_y, bn_y, t, x, y, t_fine)[0]) / (2 * epsilon)
        grad_b0y = (cost_function(b0_x, bn_x, b0_y + epsilon, bn_y, t, x, y, t_fine)[0] -
                    cost_function(b0_x, bn_x, b0_y - epsilon, bn_y, t, x, y, t_fine)[0]) / (2 * epsilon)
        grad_bny = (cost_function(b0_x, bn_x, b0_y, bn_y + epsilon, t, x, y, t_fine)[0] -
                    cost_function(b0_x, bn_x, b0_y, bn_y - epsilon, t, x, y, t_fine)[0]) / (2 * epsilon)

        # Update
        b0_x -= lr * grad_b0x
        bn_x -= lr * grad_bnx
        b0_y -= lr * grad_b0y
        bn_y -= lr * grad_bny

        if i % 10 == 0 or i == iterations - 1:
            print(f"Iter {i}: Cost = {cost:.6f} | b0_x={b0_x:.4f}, bn_x={bn_x:.4f}, b0_y={b0_y:.4f}, bn_y={bn_y:.4f}")
    
    _, x_opt, y_opt = cost_function(b0_x, bn_x, b0_y, bn_y, t, x, y, t_fine)
    return b0_x, bn_x, b0_y, bn_y, cost_history, x_opt, y_opt

# 6. Run optimization
t_fine = np.linspace(t[0], t[-1], 200)
b0_x, bn_x, b0_y, bn_y, cost_hist, x_opt, y_opt = gradient_descent(t, x, y, t_fine, lr=0.05, iterations=100)

# 7. Generate natural spline (unoptimized)
cs_x_nat = CubicSpline(t, x, bc_type='natural')
cs_y_nat = CubicSpline(t, y, bc_type='natural')
x_nat = cs_x_nat(t_fine)
y_nat = cs_y_nat(t_fine)

# 8. Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro', label='Waypoints')
plt.plot(x_nat, y_nat, 'g--', label='Unoptimized Spline (Natural)')
plt.plot(x_opt, y_opt, 'b-', label='Optimized Spline (Min Curvature)')
plt.title("Optimized vs Unoptimized Cubic Spline")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

# 9. Plot cost vs iteration
plt.figure()
plt.plot(cost_hist, color='purple')
plt.title("Cost (∫ Curvature²) vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.show()
