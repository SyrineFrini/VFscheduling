import gurobipy as gp
from gurobipy import GRB

def solve_crop_optimization(I, R, P, T, H, theta, W, A, Q, C, S, F, Z, X):
    # Create a new model
    model = gp.Model("crop_optimization")

    # Define the decision variables
    X_irpt = {}
    for i in range(I):
        for r in range(R):
            for p in range(P):
                for t in range(T):
                    X_irpt[i, r, p, t] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{r}_{p}_{t}")

    # Set the objective function
    objective = gp.quicksum(X_irpt[i, r, p, t] * W[r][p] * A[i][t] * Q[i]
                            for i in range(I) for r in range(R) for p in range(P) for t in range(T))
    model.setObjective(objective, GRB.MAXIMIZE)

    # Add constraint 1
    for i in range(I):
        for t in range(T):
            lhs = gp.quicksum(X_irpt[i, r, p, t] * W[r][p] for r in range(R) for p in range(P))
            rhs = gp.quicksum(W[r][p] for r in range(R) for p in range(P))
            model.addConstr(lhs <= rhs)

    # Add constraint 2
    for t in range(T):
        for r in range(R):
            for p in range(P):
                if t > 0:
                    lhs1 = gp.quicksum(X_irpt[i, r, p, t - z] for i in range(I) for z in range(theta[i]))
                    model.addConstr(lhs1 <= 1)
                else:
                    lhs2 = gp.quicksum(X_irpt[i, r, p, t - z + T] for i in range(I) for z in range(theta[i - 1]))
                    model.addConstr(lhs2 <= 1)

    # Optimize the model
    model.optimize()

    # Retrieve the optimal solution
    solution = {}
    if model.status == GRB.OPTIMAL:
        for i in range(I):
            for r in range(R):
                for p in range(P):
                    for t in range(T):
                        solution[f"X_{i}_{r}_{p}_{t}"] = X_irpt[i, r, p, t].x

    return solution

# Define the data
I = 2  # Total number of crops
R = 3  # Total number of shelves
P = 4  # Total number of towers
T = 5  # Total time horizon (weeks)
H = 2  # Total number of crop families
theta = [2, 3]  # Cultivation time of each crop in weeks
W = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]  # Cultivation area of each shelf in each tower
A = [[1.5, 2.0, 2.5, 3.0, 3.5], [2.0, 2.5, 3.0, 3.5, 4.0]]  # Price per kilogram of each crop at each week
Q = [0.8, 0.9]  # Harvested quantity of each crop per cultivation area
C = {1: [1], 2: [2, 3], 3: [1, 2]}  # Set of crops in each sunlight blocking category
S = {1: [1], 2: [2], 3: [1]}  # Set of shelves in each sunlight blocking category
F = {1: [1, 2], 2: [2], 3: [1]}  # Set of crops in each crop family
Z = [0.5, 0.8]  # Average height of each crop
X = [[2.0, 1.5, 3.0, 2.5], [1.0, 2.5, 1.5, 2.0], [3.0, 1.5, 2.5, 3.0]]  # Height of each shelf in each tower

# Solve the crop optimization problem
solution = solve_crop_optimization(I, R, P, T, H, theta, W, A, Q, C, S, F, Z, X)

# Print the optimal solution
for var, val in solution.items():
    print(f"{var} = {val}")
