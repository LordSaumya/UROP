import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pd.options.mode.chained_assignment = None # default='warn'
from scipy.optimize import curve_fit
from scipy.optimize import minimize as minimise
import random
from scipy.integrate import odeint

df = 

def keplerian_orbit(state, t, GM):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)

    x_dot = vx
    y_dot = vy
    z_dot = vz

    vx_dot = -GM * x / r**3
    vy_dot = -GM * y / r**3
    vz_dot = -GM * z / r**3

    return [x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot]

def project_1d(state, los_vector):
    x, y, z, vx, vy, vz = state

    # Normalise LoS vector
    los_vector = los_vector / np.linalg.norm(los_vector)

    # Calculate the projection
    projected_position = np.dot([x, y, z], los_vector)
    projected_velocity = np.dot([vx, vy, vz], los_vector)
    
    return projected_position, projected_velocity


def generate_orbit(initial_state, times, GM, los_vector):
    orbit = odeint(keplerian_orbit, initial_state, times, args=(GM,))
    projected_orbit = np.array([project_1d(state, los_vector) for state in orbit])
    return orbit, projected_orbit[:, 0], projected_orbit[:, 1] # Full orbit, projected distances, projected velocities

def cost_function(params, times, observed_distances):
    x0, y0, z0, vx0, vy0, vz0, GM, los_x, los_y, los_z = params
    initial_state = [x0, y0, z0, vx0, vy0, vz0]
    los_vector = np.array([los_x, los_y, los_z])
    
    _, modeled_distances, _ = generate_orbit(initial_state, times, GM, los_vector)
    
    return np.sum((modeled_distances - observed_distances)**2)

initial_guess = [random.uniform(-1e7, 1e7) for _ in range(10)]  # Initial guess for the parameters
bounds = [(None, None)] * 10  # Now we have 10 parameters

# result = minimise(cost_function, initial_guess, args=(df['TOA'], df['a_dist_first_corrected']),
#                   method='L-BFGS-B', bounds=bounds)

def plot_2d_orbit(orbit, los_vector, fitted_distances):
    plt.figure(figsize=(10, 10))
    
    # Plot xy projection
    plt.plot(orbit[:, 0], orbit[:, 1], label='Orbit')
    
    # Plot line of sight projection
    max_dist = np.max(np.abs(fitted_distances))
    los_x = np.array([0, los_vector[0]]) * max_dist
    los_y = np.array([0, los_vector[1]]) * max_dist
    plt.plot(los_x, los_y, 'r--', label='Line of Sight')
    
    # Plot start and end points
    plt.plot(orbit[0, 0], orbit[0, 1], 'go', label='Start')
    plt.plot(orbit[-1, 0], orbit[-1, 1], 'ro', label='End')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Projection of Orbit')
    plt.legend()
    plt.axis('equal')  # Set equal scaling
    plt.grid(True)
    
    # Print some diagnostic information
    print(f"Initial guess for x0: {initial_guess[0]}")
    print(f"Initial guess for y0: {initial_guess[1]}")
    print(f"Initial guess for vx0: {initial_guess[3]}")
    print(f"Initial guess for vy0: {initial_guess[4]}")
    print(f"Initial guess for GM: {initial_guess[6]}")
    print(f"Initial guess for los_x: {initial_guess[7]}")
    print(f"Initial guess for los_y: {initial_guess[8]}")
    print(f"Initial guess for los_z: {initial_guess[9]}")
    print(f"Optimal parameters: {result.x}")

    print(f"Orbit start: ({orbit[0, 0]}, {orbit[0, 1]})")
    print(f"Orbit end: ({orbit[-1, 0]}, {orbit[-1, 1]})")
    print(f"Orbit shape: {orbit.shape}")
    print(f"Line of sight vector: {los_vector}")
    
    plt.show()

def plot_3d_orbit(orbit, los_vector, fitted_distances):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D orbit
    ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], label='Orbit')
    
    # Plot line of sight
    max_dist = np.max(np.abs(fitted_distances))
    los_x = np.array([0, los_vector[0]]) * max_dist
    los_y = np.array([0, los_vector[1]]) * max_dist
    los_z = np.array([0, los_vector[2]]) * max_dist
    ax.plot(los_x, los_y, los_z, 'r--', label='Line of Sight')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualisation of Orbit')
    ax.legend()

    plt.show()

# optimal_params = result.x
# full_orbit, fitted_distances, fitted_velocities = generate_orbit(optimal_params[:6], df['TOA'], optimal_params[6], optimal_params[7:10])
# # Plot the results
# plot_2d_orbit(full_orbit[:, :3], optimal_params[7:10], fitted_distances)
# plot_3d_orbit(full_orbit[:, :3], optimal_params[7:10], fitted_distances)