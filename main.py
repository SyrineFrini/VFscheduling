import gurobipy as gp
from gurobipy import GRB


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

    #Constraint 5: shelves height
    for i in range(1, I + 1):
        for t in range(1, T + 1):
            for p in range(1, P + 1):
                for r in range(1, R + 1):
                    model.addConstr(Z[i-1] * X_irpt[i, p, r, t] <= G[p, r])


    #constraint 6:
    for h in range(1, H+1):
        for t in range(1, T+1):
            for (p1, r1, p2, r2) in adjacent_tuples:

                lhs7 = gp.quicksum(X_irpt[i, p1, r1, t - z] + X_irpt[i, p2, r2, t - z] for i in F[h] for z in range(0, theta[i-1]) if (t - z) > 0)
                model.addConstr(lhs7 <= 1)

                lhs7 = gp.quicksum(
                    X_irpt[i, p1, r1, t - z + T] + X_irpt[i, p2, r2, t - z + T] for i in F[h] for z in range(0, theta[i-1]) if (t - z) <= 0)
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
theta = [2, 3, 2, 1]  # Cultivation time of each crop in weeks
W = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]  # Cultivation area of each shelf in each tower
A = [[1.5, 2.0, 2.5, 3.0, 3.5], [2.0, 2.5, 3.0, 3.5, 4.0], [3.0, 4.0, 6.0, 1.0, 4.0], [0, 0, 0, 0, 0]]  # Price per kilogram of each crop at each week
Q = [0.8, 0.9, 1, 0]  # Harvested quantity of each crop per cultivation area
C = {1: [1, 2], 2: [3], 3: [4]}
S = {1: [1], 2: [2], 3: [3]}
F = {1: [1, 2], 2: [3]}
Z = [5, 7, 4, 0]  # Average height of each crop
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
