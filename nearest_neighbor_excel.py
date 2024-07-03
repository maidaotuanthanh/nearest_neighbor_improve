import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from heapq import heappush, heappop

# Load the data from the Excel file
file_path = 'input/90fruit.xlsx'
data = pd.read_excel(file_path)
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
    min_heap = []

    while len(visited) < n:
        for city in range(n):
            if city not in visited:
                heappush(min_heap, (dist_matrix[current_city, city], city))
        while min_heap:
            min_distance, nearest_city = heappop(min_heap)
            if nearest_city not in visited:
                visited.append(nearest_city)
                current_city = nearest_city
                break

    visited.append(start_city)
    return visited

def total_distance(route, dist_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += dist_matrix[route[i], route[i + 1]]
    return total

def swap_2opt(route, i, k):
    new_route = route[0:i]
    new_route.extend(reversed(route[i:k+1]))
    new_route.extend(route[k+1:])
    return new_route

def two_opt(route, dist_matrix, max_iterations=1000):
    best_route = route
    best_distance = total_distance(route, dist_matrix)
    iterations = 0
    improved = True

    while improved and iterations < max_iterations:
        improved = False
        for i in range(1, len(route) - 2):
            for k in range(i+1, len(route) - 1):
                new_route = swap_2opt(best_route, i, k)
                new_distance = total_distance(new_route, dist_matrix)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
        iterations += 1

    return best_route, best_distance

# Precompute the distance matrix
dist_matrix = distance_matrix(coordinates)

# Find the best starting point
best_route = None
min_total_distance = float('inf')
best_start_city = None

for start_city in range(len(coordinates)):
    print(f"Starting city: {start_city}")
    route = nearest_neighbor_3d(dist_matrix, start_city)
    print(f"Initial route: {route}")
    route, total_dist = two_opt(route, dist_matrix)  # Apply 2-opt to the initial route
    print(f"Optimized route: {route} with total distance: {total_dist}")
    if total_dist < min_total_distance:
        min_total_distance = total_dist
        best_route = route
        best_start_city = start_city

# Print the best starting point and the total distance
print(f"The best starting point is city {best_start_city} with a total distance of {min_total_distance}")

# Print the route
route_coordinates = coordinates[best_route]
route_coordinates_df = pd.DataFrame(route_coordinates, columns=['x', 'y', 'z'])

# Save the route to a CSV file
output_file_path = '40_excel.csv'
route_coordinates_df.to_csv(output_file_path, index=False)
print(f"The best route has been saved to {output_file_path}")

# Plot the 3D route
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(route_coordinates[:, 0], route_coordinates[:, 1], route_coordinates[:, 2], marker='o')
for i in range(len(route_coordinates)):
    ax.text(route_coordinates[i, 0], route_coordinates[i, 1], route_coordinates[i, 2], str(i))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Plot of the Optimized Route')
plt.show()
