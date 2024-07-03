import pandas as pd
import numpy as np

# Load the data
file_path = 'input/150fruit.csv'
data = pd.read_csv(file_path)
coordinates = data.values

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def nearest_neighbor_3d(coordinates, start_city):
    n = coordinates.shape[0]
    visited = [start_city]
    current_city = start_city

    while len(visited) < n:
        nearest_city = None
        min_distance = float('inf')
        for city in range(n):
            if city not in visited:
                dist = distance(coordinates[current_city], coordinates[city])
                if dist < min_distance:
                    min_distance = dist
                    nearest_city = city
        visited.append(nearest_city)
        current_city = nearest_city

    visited.append(start_city)
    return visited

def total_distance(route, coordinates):
    total = 0
    for i in range(len(route) - 1):
        total += distance(coordinates[route[i]], coordinates[route[i + 1]])
    return total

def swap_2opt(route, i, k):
    new_route = route[0:i]
    new_route.extend(reversed(route[i:k+1]))
    new_route.extend(route[k+1:])
    return new_route

def two_opt(route, coordinates):
    best_route = route
    best_distance = total_distance(route, coordinates)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for k in range(i+1, len(route) - 1):
                new_route = swap_2opt(best_route, i, k)
                new_distance = total_distance(new_route, coordinates)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True

    return best_route, best_distance

# Find the best starting point
best_route = None
min_total_distance = float('inf')
best_start_city = None

for start_city in range(len(coordinates)):
    print(f"Starting city: {start_city}")
    route = nearest_neighbor_3d(coordinates, start_city)
    print(f"Initial route: {route}")
    route, total_dist = two_opt(route, coordinates)  # Apply 2-opt to the initial route
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
output_file_path = 'nearest_neighbor_best_route_optimized150.csv'
route_coordinates_df.to_csv(output_file_path, index=False)
print(f"The best route has been saved to {output_file_path}")
