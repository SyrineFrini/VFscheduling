import pandas as pd
import numpy as np

from model.model_MILP import solve_crop_optimization
from model.heuristic_localsearch import solve_crop_optimization_heuristic
from plot.plotting_tower_content import generate_tower_content
from plot.plotting_tower_content import plot_tower_content
from plot.plot_gantt import generate_gantt_chart
from plot.plot_animation import animate_tower_content
from plot.plot_with_height import plot_tower_content_with_height

def get_adjacent_tuples(tuples):
    adjacent_tuples = []

    for (p1, r1, p2, r2) in tuples:
        # Check adjacency condition
        if (abs(p1 - p2) == 1 and r1 == r2) or (p1 == p2 and abs(r1 - r2) == 1):
            adjacent_tuples.append((p1, r1, p2, r2))

    return adjacent_tuples

# Data Processing

crops_df = pd.read_excel("./input/Crops.xlsx")
crops_df.index = np.arange(1, len(crops_df) + 1)

shelves_df = pd.read_excel("./input/shelves.xlsx")
shelves_df.index = np.arange(1, len(shelves_df) + 1)

# Define the data
# Extract the required data from the dataframes
I = crops_df.shape[0]  # Number of crops
R = shelves_df.shape[0]  # Number of shelves

P = 2  # Number of towers
T = 20 # Time horizont

theta = {}
for i in range(1, I+1):
    theta[i] = crops_df["Cultivation time"][i]

H = crops_df['Family'].to_list()
H = list(set(H))

W = {}
for p in range(1, P+1):
    W[p] = shelves_df["harvested_area"].to_list()

A = {}
for i in range(1, I+1):
    A[i] = {}
    for t in range(1, 25):
        A[i][t] = crops_df["Average Price in Winter (€/kg)"][i]
    for t in range(25, T+1):
        A[i][t] = crops_df["Average Price in Summer (€/kg)"][i]

Q = {}
for i in range(1, I+1):
    Q[i] = crops_df["Harvested kg/m2 (Average)"][i]

min_d = {}
for i in range(1, I):
    min_d = crops_df["min demand (kg)"][i]

max_d = {}
for i in range(1, I):
    max_d = crops_df["max demand (kg)"][i]


C = crops_df.groupby('Sunlight requirement').groups
S = shelves_df.groupby('sun_categories').groups
F = crops_df.groupby('Family').groups

Z = {}
for i in range(1, I+1):
    Z[i] = crops_df["Average Height (cm)"][i]

G = {}
for p in range(1, P+1):
    for r in range(1, R+1):
        G[p, r] = shelves_df["height"][r]

# Generate the set of tuples (p1, r1, p2, r2)
cartesian_product = [(p1, r1, p2, r2) for p1 in range(1, P + 1) for r1 in range(1, R + 1) for p2 in range(1, P + 1) for
                     r2 in range(1, R + 1)]

# Get the adjacent tuples
adjacent_tuples = get_adjacent_tuples(cartesian_product)

# Solve the crop optimization problem
solution, constraint_times = solve_crop_optimization(I, R, P, T, H, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples, min_d, max_d)

# Print the optimal solution
for var, val in solution.items():
    print(f"{var} = {val}")

print(constraint_times)

generate_gantt_chart(solution, I, R, P, T, theta)
# Generate tower schedules and content
tower_data = generate_tower_content(solution, I, R, P, T, theta)
# Plot the content of tower 1 at time step 5
plot_tower_content(tower_data, 1, 10, I, R)
animate_tower_content(tower_data, 2, I, R)


