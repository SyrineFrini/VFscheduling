import gurobipy as gp
from gurobipy import GRB

def solve_crop_optimization(I, R, P, T, theta, W, A, Q, S, C):
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
    objective = gp.quicksum(X_irpt[i, p, r, t] * W[p-1][r-1] * A[i-1][t-1] * Q[i-1]
                            for i in range(1, I+1) for p in range(1, P+1) for r in range(1, R+1) for t in range(1, T+1))
    model.setObjective(objective, GRB.MAXIMIZE)

    # Add constraint 1
    for i in range(1, I+1):
        for t in range(1, T+1):
            lhs = gp.quicksum(X_irpt[i, p, r, t] * W[p-1][r-1] for p in range(1, P+1) for r in range(1, R+1))
            rhs = gp.quicksum(W[p-1][r-1] for p in range(1, P+1) for r in range(1, R+1))
            model.addConstr(lhs <= rhs)

    for t in range(1, T+1):
        for p in range(1, P+1):
            for r in range(1, R+1):
                lhs = gp.quicksum(
                    X_irpt[i, p, r, t - z] for i in range(1, I+1) for z in range(0, theta[i-1]-1) if (t - z) > 0)
                model.addConstr(lhs <= 1)

                lhs = gp.quicksum(
                    X_irpt[i, p, r, (t - z + T) ] for i in range(1, I+1) for z in range(0, theta[i-1]-1) if (t - z) <= 0)
                model.addConstr(lhs <= 1)


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
theta = [2, 3, 2, 1]  # Cultivation time of each crop in weeks
W = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]  # Cultivation area of each shelf in each tower
A = [[1.5, 2.0, 2.5, 3.0, 3.5], [2.0, 2.5, 3.0, 3.5, 4.0], [3.0, 4.0, 6.0, 1.0, 4.0], [0, 0, 0, 0, 0]]  # Price per kilogram of each crop at each week
Q = [0.8, 0.9, 1, 0]  # Harvested quantity of each crop per cultivation area
C = {1: [1, 2], 2: [3], 3: [4]}
S = {1: [1], 2: [2], 3: [3]}

# Solve the crop optimization problem
solution = solve_crop_optimization(I, R, P, T, theta, W, A, Q, C, S)

# Print the optimal solution
for var, val in solution.items():
    print(f"{var} = {val}")
