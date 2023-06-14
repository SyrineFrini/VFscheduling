import pandas as pd
import numpy as np

from model.model_MILP import solve_crop_optimization
from model.heuristic_localsearch import solve_crop_optimization_heuristic
from plot.plotting_functions import plot_tower_content
from plot.plotting_functions import generate_gantt_chart

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
T = 10 # Time horizont

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
    for t in range(1, T+1):
        A[i][t] = crops_df["Average Price in Summer (€/kg)"][i]
    #for t in range(10, T+1):
    #    A[i][t] = crops_df["Average Price in Winter (€/kg)"][i]

Q = {}
for i in range(1, I+1):
    Q[i] = crops_df["Harvested kg/m2 (Average)"][i]

C = crops_df.groupby('Sunlight requirement').groups
S = shelves_df.groupby('sun_categories').groups
F = crops_df.groupby('Family').groups

C_mapping = {'high': 1, 'medium': 2, 'low': 3}
C = {C_mapping[key]: values for key, values in C.items()}

# Mapping dictionary for S
S_mapping = {'high': 1, 'medium': 2, 'low': 3}
S = {S_mapping[key]: values for key, values in S.items()}

Z = {}
for i in range(1, I+1):
    Z[i] = crops_df["Average Height (cm)"][i]

G = {}
for p in range(1, P+1):
    for z in range(1, R+1):
        G[p, z] = shelves_df["height"][z]

# Generate the set of tuples (p1, r1, p2, r2)
cartesian_product = [(p1, r1, p2, r2) for p1 in range(1, P + 1) for r1 in range(1, R + 1) for p2 in range(1, P + 1) for
                     r2 in range(1, R + 1)]

# Get the adjacent tuples
adjacent_tuples = get_adjacent_tuples(cartesian_product)

# Solve the crop optimization problem
solution = solve_crop_optimization(I, R, P, T, H, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples)

# Print the optimal solution
for var, val in solution.items():
    print(f"{var} = {val}")

generate_gantt_chart(solution, I, R, P, T, theta)
plot_tower_content(solution, I, R, P, T, 7)
