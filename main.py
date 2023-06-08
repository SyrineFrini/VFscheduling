import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import random

def generate_gantt_chart(solution, I, R, P, T, theta):
    # Create a list of crops
    crops = ["Crop {}".format(i) for i in range(1, I+1)]

    # Create a dictionary to store the job schedule for each tower
    tower_schedules = {}

    for p in range(1, P+1):
        # Initialize the schedule for the current tower
        tower_schedule = [[] for _ in range(R)]

        # Assign colors to each crop
        crop_colors = {crops[i]: random.choice(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF']) for i in range(I)}

        # Iterate over the shelves and time periods
        for r in range(1, R+1):
            for t in range(1, T+1):
                # Check if a crop is planted at the current position (p, r, t)
                for i in range(1, I+1):
                    if solution.get("X_{}_{}_{}_{}".format(i, p, r, t), 0) == 1:
                        crop_length = theta[i]
                        crop_start = ((t - 1) % T) + 1  # Adjust start time if rotation is needed
                        tower_schedule[r-1].append((crop_start, crop_length, crops[i-1], crop_colors[crops[i-1]]))

        # Store the schedule for the current tower
        tower_schedules[p] = tower_schedule

    # Plot the Gantt chart for each tower separately
    for p, tower_schedule in tower_schedules.items():
        fig, ax = plt.subplots(figsize=(10, 5))

        for r in range(R):
            for job in tower_schedule[r]:
                ax.barh(r, job[1], align='center', height=0.5, left=job[0], color=job[3])
                ax.text(job[0] + job[1] / 2, r + 0.25, job[2], ha='center', va='center')

            ax.set_yticks(range(R))
            ax.set_yticklabels(["Shelf {}".format(r+1) for r in range(R)])
            ax.set_xlabel("Time")
            ax.set_title("Tower {}".format(p))

        plt.show()

def get_adjacent_tuples(tuples):
    adjacent_tuples = []

    for (p1, r1, p2, r2) in tuples:
        # Check adjacency condition
        if (abs(p1 - p2) == 1 and r1 == r2) or (p1 == p2 and abs(r1 - r2) == 1):
            adjacent_tuples.append((p1, r1, p2, r2))

    return adjacent_tuples

def solve_crop_optimization(I, R, P, T, theta, W, A, Q, S, C, Z, G, F):
    # Create a new model
    model = gp.Model("crop_optimization")

    # Define the decision variables
    X_irpt = {}
    for i in range(1, I+1):
        for p in range(1, P+1):
            for r in range(1, R+1):
                for t in range(1, T+1):
                    X_irpt[i, p, r, t] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{p}_{r}_{t}")

    # Set the objective function
    objective = gp.quicksum(X_irpt[i, p, r, t] * W[p][r-1] * A[i][t-1] * Q[i]
                            for i in range(1, I+1) for p in range(1, P+1) for r in range(1, R+1) for t in range(1, T+1))
    model.setObjective(objective, GRB.MAXIMIZE)

    # Add constraint 1
    for i in range(1, I+1):
        for t in range(1, T+1):
            lhs = gp.quicksum(X_irpt[i, p, r, t] * W[p][r-1] for p in range(1, P+1) for r in range(1, R+1))
            rhs = gp.quicksum(W[p][r-1] for p in range(1, P+1) for r in range(1, R+1))
            model.addConstr(lhs <= rhs)

    for t in range(1, T + 1):
        for p in range(1, P + 1):
            for r in range(1, R + 1):
                lhs_1 = gp.quicksum(
                    X_irpt[i, p, r, t - z + T] if (t - z) <= 0 else X_irpt[i, p, r, t - z]
                    for i in range(1, I + 1)
                    for z in range(0, theta[i])
                )
                model.addConstr(lhs_1 <= 1)




    # Add constraint 3: Maintenance period for each shelf
    for r in range(1, R+1):
       for p in range(1, P+1):
            lhs3 = gp.quicksum(X_irpt[4, p, r, t] for t in range(1, T+1))
            model.addConstr(lhs3 == 1)

    # Add Constraint 4: Sun categories
    for t in range(1,T+1):
        for p in range(1, P+1):
            lhs4 = gp.quicksum(X_irpt[i, p, r, t] for i in C[1] for r in S[2]) * gp.quicksum(X_irpt[i, p, r, t] for i in C[1] for r in S[3])
            lhs5 = gp.quicksum(X_irpt[i, p, r, t] for i in C[2] for r in S[3])
            lhs6 = gp.quicksum(X_irpt[i, p, r, t] for i in C[3] for r in S[1])
            model.addConstr(lhs4 == 0)
            model.addConstr(lhs5 == 0)
            model.addConstr(lhs6 == 0)

    #Constraint 5: shelves height
    for i in range(1, I + 1):
        for t in range(1, T + 1):
            for p in range(1, P + 1):
                for r in range(1, R + 1):
                    model.addConstr(Z[i] * X_irpt[i, p, r, t] <= G[p, r])


    #constraint 6:
    for h in range(1, H+1):
        for t in range(1, T+1):
            for (p1, r1, p2, r2) in adjacent_tuples:
                lhs7 = gp.quicksum(
                            X_irpt[i, p1, r1, t - z + T] + X_irpt[i, p2, r2, t - z + T]  if (t - z) <= 0 else X_irpt[i, p1, r1, t - z ] + X_irpt[i, p2, r2, t - z]
                            for i in F[h]
                            for z in range(0, theta[i]+1))
                model.addConstr(lhs7 <= 1)



    # Optimize the model
    model.optimize()

    # Retrieve the optimal solution
    solution = {}
    if model.status == GRB.OPTIMAL:
        for i in range(1, I+1):
            for p in range(1, P+1):
                for r in range(1, R+1):
                    for t in range(1, T+1):
                        solution[f"X_{i}_{p}_{r}_{t}"] = X_irpt[i, p, r, t].x


    return solution

# Define the data
I = 4  # Total number of crops // the fourth one is MAINTENANCE
R = 4  # Total number of shelves
P = 3  # Total number of towers
T = 5  # Total time horizon (weeks)
H = 2  # Total number of crop families
theta  = {1: 2, 2: 3, 3: 2, 4:1}  # Cultivation time of each crop in weeks
W = {1: [1, 2, 3, 4], 2:[2, 3, 4, 5], 3:[3, 4, 5, 6]}  # Cultivation area of each shelf in each tower
A = {1: [1.5, 2.0, 2.5, 3.0, 3.5], 2: [2.0, 2.5, 3.0, 3.5, 4.0], 3: [3.0, 4.0, 6.0, 1.0, 4.0], 4:[0, 0, 0, 0, 0]}  # Price per kilogram of each crop at each week
Q = {1:0.8, 2:0.9, 3:1, 4:0}  # Harvested quantity of each crop per cultivation area
C = {1: [1, 2], 2: [3], 3: [4]} #Sun categories of the selves
S = {1: [1], 2: [2], 3: [3]} #sun categories of the plants
F = {1: [1, 2], 2: [3]} #Families of the plant
Z = {1:5, 2:7, 3:4, 4:0}  # Average height of each crop
G = {(1, 1): 10, (1, 2): 5, (1, 3): 8, (1, 4): 12, (2, 1): 7, (2, 2): 9, (2, 3): 6, (2, 4): 11, (3, 1): 9, (3, 2): 6, (3, 3): 10, (3, 4): 8}

# Generate the set of tuples (p1, r1, p2, r2)
cartesian_product = [(p1, r1, p2, r2) for p1 in range(1, P+1) for r1 in range(1, R+1) for p2 in range(1, P+1) for r2 in range(1, R+1)]

# Get the adjacent tuples
adjacent_tuples = get_adjacent_tuples(cartesian_product)
print(adjacent_tuples)

# Solve the crop optimization problem
solution = solve_crop_optimization(I, R, P, T, theta, W, A, Q, C, S, Z, G, F)

# Print the optimal solution
for var, val in solution.items():
    print(f"{var} = {val}")

generate_gantt_chart(solution, I, R, P, T, theta)
