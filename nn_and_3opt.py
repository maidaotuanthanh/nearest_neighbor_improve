import pandas as pd
import numpy as np

# Load the data
file_path = 'input/40fruit.csv'
data = pd.read_csv(file_path)
coordinates = data.values

def distance_matrix(coordinates):
    n = coordinates.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def nearest_neighbor_3d(dist_matrix, start_city):
    n = dist_matrix.shape[0]
    visited = [start_city]
    current_city = start_city

    while len(visited) < n:
        nearest_city = None
        min_distance = float('inf')
        for city in range(n):
            if city not in visited:
                dist = dist_matrix[current_city, city]
                if dist < min_distance:
                    min_distance = dist
                    nearest_city = city
        visited.append(nearest_city)
        current_city = nearest_city

    visited.append(start_city)
    return visited

def total_distance(route, dist_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += dist_matrix[route[i], route[i + 1]]
    total += dist_matrix[route[-1], route[0]]
    return total

def swap_2opt(route, i, k):
    new_route = route[0:i]
    new_route.extend(reversed(route[i:k+1]))
    new_route.extend(route[k+1:])
    return new_route

def two_opt(route, dist_matrix):
    best_route = route
    best_distance = total_distance(route, dist_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for k in range(i+1, len(route) - 1):
                new_route = swap_2opt(best_route, i, k)
                new_distance = total_distance(new_route, dist_matrix)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True

    return best_route, best_distance

def swap_3opt(route, i, j, k):
    new_routes = []
    new_routes.append(route[:i] + route[i:j][::-1] + route[j:k][::-1] + route[k:])
    new_routes.append(route[:i] + route[i:j] + route[j:k][::-1] + route[k:])
    new_routes.append(route[:i] + route[i:j][::-1] + route[j:k] + route[k:])
    new_routes.append(route[:i] + route[j:k] + route[i:j] + route[k:])
    new_routes.append(route[:i] + route[j:k][::-1] + route[i:j] + route[k:])
    new_routes.append(route[:i] + route[k:] + route[j:k] + route[i:j])
    new_routes.append(route[:i] + route[k:] + route[i:j][::-1] + route[j:k][::-1])
    return new_routes

def three_opt(route, dist_matrix):
    best_route = route
    best_distance = total_distance(route, dist_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 3):
            for j in range(i + 1, len(route) - 2):
                for k in range(j + 1, len(route) - 1):
                    new_routes = swap_3opt(best_route, i, j, k)
                    for new_route in new_routes:
                        new_distance = total_distance(new_route, dist_matrix)
                        if new_distance < best_distance:
                            best_route = new_route
                            best_distance = new_distance
                            improved = True

    return best_route, best_distance

# Precompute the distance matrix
dist_matrix = distance_matrix(coordinates)

# Find the best starting point
best_route_2opt = None
min_total_distance_2opt = float('inf')
best_route_3opt = None
min_total_distance_3opt = float('inf')
best_start_city = None

for start_city in range(len(coordinates)):
    print(f"Starting city: {start_city}")
    route = nearest_neighbor_3d(dist_matrix, start_city)
    print(f"Initial route: {route}")

    # Apply 2-opt to the initial route
    route_2opt, total_dist_2opt = two_opt(route, dist_matrix)
    print(f"Optimized route with 2-opt: {route_2opt} with total distance: {total_dist_2opt}")
    if total_dist_2opt < min_total_distance_2opt:
        min_total_distance_2opt = total_dist_2opt
        best_route_2opt = route_2opt

    # Apply 3-opt to the initial route
    route_3opt, total_dist_3opt = three_opt(route, dist_matrix)
    print(f"Optimized route with 3-opt: {route_3opt} with total distance: {total_dist_3opt}")
    if total_dist_3opt < min_total_distance_3opt:
        min_total_distance_3opt = total_dist_3opt
        best_route_3opt = route_3opt

# Print the best results for 2-opt and 3-opt
print(f"The best total distance with 2-opt is {min_total_distance_2opt}")
print(f"The best total distance with 3-opt is {min_total_distance_3opt}")

# Save the best routes to CSV files
route_coordinates_2opt = coordinates[best_route_2opt]
route_coordinates_df_2opt = pd.DataFrame(route_coordinates_2opt, columns=['x', 'y', 'z'])
output_file_path_2opt = 'output/nearest_neighbor_best_route_optimized_2opt.csv'
route_coordinates_df_2opt.to_csv(output_file_path_2opt, index=False)
print(f"The best route with 2-opt has been saved to {output_file_path_2opt}")

route_coordinates_3opt = coordinates[best_route_3opt]
route_coordinates_df_3opt = pd.DataFrame(route_coordinates_3opt, columns=['x', 'y', 'z'])
output_file_path_3opt = 'output/nearest_neighbor_best_route_optimized_3opt.csv'
route_coordinates_df_3opt.to_csv(output_file_path_3opt, index=False)
print(f"The best route with 3-opt has been saved to {output_file_path_3opt}")

# Plot the 3D routes for 2-opt and 3-opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 6))

# 2-opt route
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(route_coordinates_2opt[:, 0], route_coordinates_2opt[:, 1], route_coordinates_2opt[:, 2], marker='o')
for i in range(len(route_coordinates_2opt)):
    ax1.text(route_coordinates_2opt[i, 0], route_coordinates_2opt[i, 1], route_coordinates_2opt[i, 2], str(i))
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Plot of the Optimized Route with 2-opt')

# 3-opt route
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(route_coordinates_3opt[:, 0], route_coordinates_3opt[:, 1], route_coordinates_3opt[:, 2], marker='o')
for i in range(len(route_coordinates_3opt)):
    ax2.text(route_coordinates_3opt[i, 0], route_coordinates_3opt[i, 1], route_coordinates_3opt[i, 2], str(i))
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('3D Plot of the Optimized Route with 3-opt')

plt.show()
