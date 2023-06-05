import gurobipy as gp
from gurobipy import GRB

def solve_crop_optimization(I, R, P, T, theta, W, A, Q):
    # Create a new model
    model = gp.Model("crop_optimization")

    # Define the decision variables
    X_irpt = {}
    for i in range(I):
        for p in range(P):
            for r in range(R):
                for t in range(T):
                    X_irpt[i, p, r, t] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{p}_{r}_{t}")

    # Set the objective function
    objective = gp.quicksum(X_irpt[i, p, r, t] * W[p][r] * A[i][t] * Q[i]
                            for i in range(I) for p in range(P) for r in range(R) for t in range(T))
    model.setObjective(objective, GRB.MAXIMIZE)

    # Add constraint 1
    for i in range(I):
        for t in range(T):
            lhs = gp.quicksum(X_irpt[i, p, r, t] * W[p][r] for p in range(P) for r in range(R))
            rhs = gp.quicksum(W[p][r] for p in range(P) for r in range(R))
            model.addConstr(lhs <= rhs)

    # Add constraint 2
    for t in range(T):
        for p in range(P):
            for r in range(R):
                if t > 0:
                    lhs1 = gp.quicksum(X_irpt[i, p, r, t - z] for i in range(I) for z in range(theta[i]-1))
                    model.addConstr(lhs1 <= 1)
                else:
                    lhs2 = gp.quicksum(X_irpt[i, p, r, (t - z + T) % T] for i in range(I) for z in range(theta[i]-1))
                    model.addConstr(lhs2 <= 1)

    # Optimize the model
    model.optimize()

    # Retrieve the optimal solution
    solution = {}
    if model.status == GRB.OPTIMAL:
        for i in range(I):
            for p in range(P):
                for r in range(R):
                    for t in range(T):
                        solution[f"X_{i}_{p}_{r}_{t}"] = X_irpt[i, p, r, t].x

    return solution

# Define the data
I = 2  # Total number of crops
R = 4  # Total number of shelves
P = 3  # Total number of towers
T = 5  # Total time horizon (weeks)
H = 2  # Total number of crop families
theta = [2, 3]  # Cultivation time of each crop in weeks
W = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]  # Cultivation area of each shelf in each tower
A = [[1.5, 2.0, 2.5, 3.0, 3.5], [2.0, 2.5, 3.0, 3.5, 4.0]]  # Price per kilogram of each crop at each week
Q = [0.8, 0.9]  # Harvested quantity of each crop per cultivation area

# Solve the crop optimization problem
solution = solve_crop_optimization(I, R, P, T, theta, W, A, Q)

# Print the optimal solution
for var, val in solution.items():
    print(f"{var} = {val}")
