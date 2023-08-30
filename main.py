import pandas as pd
import numpy as np
import time
import random

from model.model_MILP import solve_crop_optimization
from model.heuristic_localsearch import solve_crop_optimization_heuristic
from plot.plotting_tower_content import generate_tower_content
from plot.plotting_tower_content import plot_tower_content
from plot.plot_gantt import generate_gantt_chart
from plot.plot_animation import animate_tower_content
from plot.plot_with_height import plot_tower_content_with_height


def time_sensitivity_analysis(P, I, R, H, theta, W, A, Q, C, S, Z, G, F, min_d, max_d):
    time_periods = [20, 30, 40]  # List of time periods for sensitivity analysis
    results = []

    for T in time_periods:
        # Update any other parameters that depend on T if necessary
        # Generate the set of tuples (p1, r1, p2, r2)
        cartesian_product = [(p1, r1, p2, r2) for p1 in range(1, P + 1) for r1 in range(1, R + 1) for p2 in range(1, P + 1) for r2 in range(1, R + 1)]

        # Get the adjacent tuples
        adjacent_tuples = get_adjacent_tuples(cartesian_product)
        D = generate_demand_dict(crops_df, T, P)
        print(D)

        # Call your optimization function
        solution, constraint_times, model, revenue = solve_crop_optimization(I, R, P, T, H, D, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples, min_d, max_d)
        generate_gantt_chart(solution, I, R, P, T, theta)

        # Retrieve the gap, runtime and the objective value from the model
        gap = model.MIPGap
        objective_value = model.ObjVal
        run_time = model.Runtime

        # Store the results
        results.append({
            'T': T,
            'solution': solution,
            'run_time': run_time,
            'gap': gap,
            'objective_value': objective_value,
            'revenue': revenue
        })

    # Here, you can save results to a file or return them for further analysis
    return results
def towers_sensitivity_analysis(min_towers, max_towers, step, I, R, T, H, theta, W, A, Q, C, S, Z, G, F, min_d, max_d):
    results = []

    for P in range(min_towers, max_towers + 1, step):
        # Update any other parameters that depend on P if necessary
        # Generate the set of tuples (p1, r1, p2, r2)
        cartesian_product = [(p1, r1, p2, r2) for p1 in range(1, P + 1) for r1 in range(1, R + 1) for p2 in
                             range(1, P + 1) for
                             r2 in range(1, R + 1)]

        # Get the adjacent tuples
        adjacent_tuples = get_adjacent_tuples(cartesian_product)
        D = generate_demand_dict(crops_df, T, P)
        print(D)
        # Call your optimization function
        solution, constraint_times, model, revenue = solve_crop_optimization(I, R, P, T, H, D, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples, min_d, max_d)

        # Retrieve the gap and the objective value from the model
        gap = model.MIPGap
        objective_value = model.ObjVal
        run_time = model.Runtime

        # Store the results
        results.append({
            'P': P,
            'solution': solution,
            'run_time': run_time,
            'gap': gap,
            'objective_value': objective_value,
            'revenue': revenue
        })

    # Here, you can save results to a file or return them for further analysis
    return results



def get_adjacent_tuples(tuples):
    adjacent_tuples = []

    for (p1, r1, p2, r2) in tuples:
        # Check adjacency condition
        if (abs(p1 - p2) == 1 and r1 == r2) or (p1 == p2 and abs(r1 - r2) == 1):
            adjacent_tuples.append((p1, r1, p2, r2))

    return adjacent_tuples

def generate_demand_dict(crops_df, T, num_towers):
    demand_dict = {}
    random.seed(0)

    for t in range(1, T + 1):
        demand_dict[t] = {}
        for i, row in crops_df.iterrows():
            demand = random.uniform(0, 5) * num_towers
            demand_dict[t][i] = demand
    return demand_dict


# Data Processing
crops_df = pd.read_excel("./input/Crops_12.xlsx")
crops_df.index = np.arange(1, len(crops_df) + 1)

shelves_df = pd.read_excel("./input/shelves.xlsx")
shelves_df.index = np.arange(1, len(shelves_df) + 1)

# Define the data
# Extract the required data from the dataframes
I = crops_df.shape[0]  # Number of crops
R = shelves_df.shape[0]  # Number of shelves

P = 2  # Number of towers
T = 30 # Time horizont

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

# Sample usage:
# Assuming crops_df is your DataFrame, T is the number of weeks, and num_towers is the number of towers
D = generate_demand_dict(crops_df, T, P)

# Generate the set of tuples (p1, r1, p2, r2)
cartesian_product = [(p1, r1, p2, r2) for p1 in range(1, P + 1) for r1 in range(1, R + 1) for p2 in range(1, P + 1) for
                     r2 in range(1, R + 1)]

# Get the adjacent tuples
adjacent_tuples = get_adjacent_tuples(cartesian_product)

D = generate_demand_dict(crops_df, T, P)
# Solve the crop optimization problem
solution, constraint_times, model, revenue = solve_crop_optimization(I, R, P, T, H, D, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples, min_d, max_d)

results = []
# Retrieve the gap and the objective value from the model
gap = model.MIPGap
objective_value = model.ObjVal
run_time = model.Runtime

# Store the results
results.append({
    'P': P,
    'solution': solution,
    'run_time': run_time,
    'gap': gap,
    'objective_value': objective_value,
    'revenue': revenue
})

#Print the optimal solution
for var, val in solution.items():
    print(f"{var} = {val}")


generate_gantt_chart(solution, I, R, P, T, theta)
# Generate tower schedules and content
#tower_data = generate_tower_content(solution, I, R, P, T, theta)
# Plot the content of tower 1 at time step 5
#plot_tower_content(tower_data, 1, 10, I, R)
#animate_tower_content(tower_data, 2, I, R)


min_tower_value = 2
max_tower_value = 4
step_size = 1

#results_T = time_sensitivity_analysis(P, I, R, H, theta, W, A, Q, C, S, Z, G, F, min_d, max_d)'''
#print(results)
