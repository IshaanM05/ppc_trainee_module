import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def cubic_interp1d(x0, x, y):
    """
    Interpolate a 1-D function using cubic splines.
    x0 : a float or an 1d-array
    x : (N,) array_like
        A 1-D array of real/complex values.
    y : (N,) array_like
        A 1-D array of real values. The length of y along the
        interpolation axis must be equal to the length of x.

    Implement a trick to generate at first step the cholesky matrice L of
    the tridiagonal matrice A (thus L is a bidiagonal matrice that
    can be solved in two distinct loops).

    additional ref: www.math.uh.edu/~jingqiu/math4364/spline.pdf
    """
    x = np.asfarray(x)  # Ensure x is a float array
    y = np.asfarray(y)  # Ensure y is a float array

    # Check if x is sorted. If not, sort it.
    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)  # Number of data points

    # Calculate the differences between consecutive x and y values
    xdiff = np.diff(x)
    ydiff = np.diff(y)

    # Allocate buffer arrays for the Li and Li-1 diagonals, and z (the spline coefficients)
    Li = np.empty(size)
    Li_1 = np.empty(size-1)
    z = np.empty(size)

    # Initialize the first diagonal value and boundary condition for the natural spline
    Li[0] = sqrt(2*xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0  # Natural boundary condition (second derivative at the endpoints is 0)
    z[0] = B0 / Li[0]

    # Loop to fill in the diagonals and solve the system [L][y] = [B]
    for i in range(1, size-1):
        Li_1[i] = xdiff[i-1] / Li[i-1]  # Compute Li-1 (subdiagonal)
        term = 2 * (xdiff[i-1] + xdiff[i]) - Li_1[i-1]**2
        Li[i] = sqrt(term) if term >= 0 else 0.0001  # Avoid negative square root
        
        Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i-1] / xdiff[i-1])  # Compute B (right-hand side)
        z[i] = (Bi - Li_1[i-1] * z[i-1]) / Li[i]  # Solve for z (second derivative)

    # Handle the last point, for which the boundary condition is natural (second derivative = 0)
    i = size - 1
    Li_1[i-1] = xdiff[-1] / Li[i-1]
    term = 2 * xdiff[-1] - Li_1[i-1]**2
    Li[i] = sqrt(term) if term >= 0 else 0.0001
    Bi = 0.0  # Natural boundary condition
    z[i] = (Bi - Li_1[i-1] * z[i-1]) / Li[i]

    # Solve the system [L.T][z] = [y] to get the second derivatives (z)
    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i] * z[i+1]) / Li[i]

    # Find the index of the nearest x0 in x (for interpolation)
    index = x.searchsorted(x0)
    index = np.clip(index, 1, size-1)  # Ensure index is within bounds

    xi1, xi0 = x[index], x[index-1]  # x values for interpolation
    yi1, yi0 = y[index], y[index-1]  # y values for interpolation
    zi1, zi0 = z[index], z[index-1]  # Second derivatives for interpolation
    hi1 = xi1 - xi0  # Distance between the two x values

    # Perform the cubic interpolation
    f0 = zi0 / (6 * hi1) * (xi1 - x0)**3 + \
         zi1 / (6 * hi1) * (x0 - xi0)**3 + \
         (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0) + \
         (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)
    return f0

# Main function to plot the cubic spline interpolation
if __name__ == '__main__':
    # Generate 4 random data points for interpolation
    # Ask the user to enter a value for n
    n = int(input("Please enter the value for n: "))
    print(f"The value of n is: {n}")

    x = np.sort(np.random.rand(n) * 100)  # Random x values between 0 and 100
    y = np.sin(x)  # Corresponding y values using the sine function

    # Plot the original points
    plt.scatter(x, y, color='red', label='Data Points')

    # Create a fine grid of x values for interpolation
    x_new = np.linspace(0, 100, 2010)  # 201 points between 0 and 100 for smoothness
    # Plot the cubic spline interpolation
    y_new = np.array([cubic_interp1d(x0, x, y) for x0 in x_new])  # Perform cubic spline interpolation
    plt.plot(x_new, y_new, label='Cubic Spline Interpolation')

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # Show the plot
    plt.show()
