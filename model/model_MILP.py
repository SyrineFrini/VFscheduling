import gurobipy as gp
from gurobipy import GRB

def solve_crop_optimization(I, R, P, T, H, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples):

    # Create a new model
    model = gp.Model("crop_optimization")

    # Define the decision variables
    X_irpt = {}
    for i in range(1, I + 1):
        for p in range(1, P + 1):
            for r in range(1, R + 1):
                for t in range(1, T + 1):
                    X_irpt[i, p, r, t] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{p}_{r}_{t}")

    # Set the objective function
    objective = gp.quicksum(X_irpt[i, p, r, t] * W[p][r - 1] * A[i][t] * Q[i]
                            for i in range(1, I + 1) for p in range(1, P + 1) for r in range(1, R + 1)
                            for t in range(1, T + 1))
    model.setObjective(objective, GRB.MAXIMIZE)

    # Add constraint 1
    for i in range(1, I + 1):
        for t in range(1, T + 1):
            lhs = gp.quicksum(X_irpt[i, p, r, t] * W[p][r - 1] for p in range(1, P + 1) for r in range(1, R + 1))
            rhs = gp.quicksum(W[p][r - 1] for p in range(1, P + 1) for r in range(1, R + 1))
            model.addConstr(lhs <= rhs)

    for i in range(1, I + 1):
        lhs_3 = gp.quicksum(X_irpt[i, p, r, t] * W[p][r - 1] * Q[i] for p in range(1, P + 1) for r in range(1, R + 1) for t in range(1, T+1))
        model.addConstr(lhs_3 <= 50)

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
    for r in range(1, R + 1):
        for p in range(1, P + 1):
            lhs3 = gp.quicksum(X_irpt[I, p, r, t] for t in range(1, T + 1))
            model.addConstr(lhs3 == 1)

    # Add Constraint 4: Sun categories
    for t in range(1, T + 1):
        for p in range(1, P + 1):
            lhs4 = gp.quicksum(X_irpt[i, p, r, t] for i in C[1] for r in S[2]) * gp.quicksum(X_irpt[i, p, r, t] for i in C[1] for r in S[3])
            lhs5 = gp.quicksum(X_irpt[i, p, r, t] for i in C[2] for r in S[3])
            lhs6 = gp.quicksum(X_irpt[i, p, r, t] for i in C[3] for r in S[1])
            model.addConstr(lhs4 == 0)
            model.addConstr(lhs5 == 0)
            model.addConstr(lhs6 == 0)

    # Constraint 5: shelves height
    for i in range(1, I + 1):
        for t in range(1, T + 1):
            for p in range(1, P + 1):
                for r in range(1, R + 1):
                    model.addConstr(Z[i] * X_irpt[i, p, r, t] <= G[p, r])

    # constraint 6:
    for h in F:
        for t in range(1, T + 1):
            for p in range(1, P + 1):
                for r in range(1, R + 1):
                    lhs7 = gp.quicksum(
                        X_irpt[i, p, r, t - z + T] if (t - z) <= 0 else X_irpt[i, p, r, t - z]
                            for i in F[h]
                            for z in range(0, theta[i]+1))
                    model.addConstr(lhs7 <= 1)

    '''#constraint 7
    for h in F:
        for t in range(1, T + 1):
            for (p1, r1, p2, r2) in adjacent_tuples:
                lhs8 = gp.quicksum(
                    X_irpt[i, p1, r1, t - z + T] + X_irpt[i, p2, r2, t - z + T] if (t - z) <= 0 else X_irpt[
                                                                                                         i, p1, r1, t - z] +
                                                                                                     X_irpt[
                                                                                                         i, p2, r2, t - z]
                    for i in F[h]
                    for z in range(0, theta[i]))
                model.addConstr(lhs8 <= 1)'''


    #constraint 8
    for h in F:
        for t in range(1, T + 1):
            for p in range(1, P + 1):
                for r in range(2, R + 1):
                    lhs9 = gp.quicksum(
                        X_irpt[i, p, r, t - z + T] + X_irpt[i, p, r - 1, t - z + T] if (t - z) <= 0 else X_irpt[
                                                                                                             i, p, r, t - z] +
                                                                                                         X_irpt[
                                                                                                             i, p, r - 1, t - z]
                        for i in F[h]
                        for z in range(0, theta[i]))
                    model.addConstr(lhs9 <= 1)

    # Optimize the model
    model.optimize()

    # Retrieve the optimal solution
    solution = {}
    if model.status == GRB.OPTIMAL:
        for i in range(1, I + 1):
            for p in range(1, P + 1):
                for r in range(1, R + 1):
                    for t in range(1, T + 1):
                        solution[f"X_{i}_{p}_{r}_{t}"] = X_irpt[i, p, r, t].x

    return solution
