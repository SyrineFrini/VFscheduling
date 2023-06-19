import gurobipy as gp
from gurobipy import GRB
import time

def solve_crop_optimization(I, R, P, T, H, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples, min_d, max_d):

    # Create a new model
    model = gp.Model("crop_optimization")

    # Define the decision variables
    X_irpt = {}
    constraint_times = {}
    for i in range(1, I + 1):
        for p in range(1, P + 1):
            for r in range(1, R + 1):
                for t in range(1, T + 1):
                    X_irpt[i, p, r, t] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{p}_{r}_{t}")

    # Set the objective function
    objective = gp.quicksum(X_irpt[i, p, r, t] * W[p][r - 1] * ((t + theta[i] - 1) % T + 1) * Q[i]
                            for i in range(1, I + 1) for p in range(1, P + 1) for r in range(1, R + 1)
                            for t in range(1, T + 1))
    model.setObjective(objective, GRB.MAXIMIZE)

    # Add constraint 1
    start_time = time.time()
    for i in range(1, I + 1):
        for t in range(1, T + 1):
            lhs = gp.quicksum(X_irpt[i, p, r, t] * W[p][r - 1] for p in range(1, P + 1) for r in range(1, R + 1))
            rhs = gp.quicksum(W[p][r - 1] for p in range(1, P + 1) for r in range(1, R + 1))
            model.addConstr(lhs <= rhs)
    end_time = time.time()
    constraint_times["Constraint 1"] = end_time - start_time

    # Add constraint 2
    start_time = time.time()
    for t in range(1, T + 1):
        for p in range(1, P + 1):
            for r in range(1, R + 1):
                lhs_1 = gp.quicksum(
                    X_irpt[i, p, r, t - z + T] if (t - z) <= 0 else X_irpt[i, p, r, t - z]
                    for i in range(1, I + 1)
                    for z in range(0, theta[i])
                )
                model.addConstr(lhs_1 <= 1)
    end_time = time.time()
    constraint_times["Constraint 2"] = end_time - start_time

    # Add constraint 3
    start_time = time.time()
    for r in range(1, R + 1):
        for p in range(1, P + 1):
            lhs3 = gp.quicksum(X_irpt[I, p, r, t] for t in range(1, T + 1))
            model.addConstr(lhs3 == 1)
    end_time = time.time()
    constraint_times["Constraint 3"] = end_time - start_time

    # Add Constraint 4
    start_time = time.time()
    for t in range(1, T + 1):
        for p in range(1, P + 1):
            lhs4 = gp.quicksum(X_irpt[i, p, r, t] for i in C["high"] for r in S["medium"]) * gp.quicksum(
                X_irpt[i, p, r, t] for i in C["high"] for r in S["low"])
            lhs5 = gp.quicksum(X_irpt[i, p, r, t] for i in C["medium"] for r in S["low"])
            lhs6 = gp.quicksum(X_irpt[i, p, r, t] for i in C["low"] for r in S["high"])
            model.addConstr(lhs4 == 0)
            model.addConstr(lhs5 == 0)
            model.addConstr(lhs6 == 0)
    end_time = time.time()
    constraint_times["Constraint 4"] = end_time - start_time

    # Constraint 5
    start_time = time.time()
    for i in range(1, I + 1):
        for t in range(1, T + 1):
            for p in range(1, P + 1):
                for r in range(1, R + 1):
                    model.addConstr(Z[i] * X_irpt[i, p, r, t] <= G[p, r])
    end_time = time.time()
    constraint_times["Constraint 5"] = end_time - start_time

    # Constraint 6
    start_time = time.time()
    for h in F:
        for t in range(1, T + 1):
            for p in range(1, P + 1):
                for r in range(1, R + 1):
                    lhs7 = gp.quicksum(
                        X_irpt[i, p, r, t - z + T] if (t - z) <= 0 else X_irpt[i, p, r, t - z]
                        for i in F[h]
                        for z in range(0, theta[i] + 1))
                    model.addConstr(lhs7 <= 1)
    end_time = time.time()
    constraint_times["Constraint 6"] = end_time - start_time

    # Constraint 7
    start_time = time.time()
    '''for h in F:
        for t in range(1, T + 1):
            for (p1, r1, p2, r2) in adjacent_tuples:
                lhs8 = gp.quicksum(
                    X_irpt[i, p1, r1, t - z + T] + X_irpt[i, p2, r2, t - z + T] if (t - z) <= 0 else X_irpt[
                        i, p1, r1, t - z] +
                                                                                             X_irpt[
                                                                                                 i, p2, r2, t - z]
                    for i in F[h]
                    for z in range(0, theta[i]))
                model.addConstr(lhs8 <= 1)
    end_time = time.time()'''
    constraint_times["Constraint 7"] = end_time - start_time

    # Constraint 8
    start_time = time.time()
    for h in F:
        for t in range(1, T + 1):
            for p in range(1, P + 1):
                for r in range(2, R + 1):
                    lhs9 = gp.quicksum(
                        X_irpt[i, p, r, t - z + T] + X_irpt[i, p, r - 1, t - z + T] if (t - z) <= 0 else X_irpt[
                            i, p, r, t - z] +
                                                                                                 X_irpt[
                                                                                                     i, p, r - 1,
                                                                                                     t - z]
                        for i in F[h]
                        for z in range(0, theta[i]))
                    model.addConstr(lhs9 <= 1)
    end_time = time.time()
    constraint_times["Constraint 8"] = end_time - start_time

    for i in range(1, I):
        lhs10 = gp.quicksum(X_irpt[i, p, r, t] * W[p][r - 1] * Q[i] for p in range(1, P + 1) for r in range(1, R + 1) for t in range(1, T+1))
        model.addConstr(lhs10 <= max_d)
        model.addConstr(lhs10 >= min_d)


    # Optimize the model
    model.setParam('MIPGap', 0.01)
    model.optimize()

    # Retrieve the optimal solution
    solution = {}
    if model.status == GRB.OPTIMAL:
        for i in range(1, I + 1):
            for p in range(1, P + 1):
                for r in range(1, R + 1):
                    for t in range(1, T + 1):
                        solution[f"X_{i}_{p}_{r}_{t}"] = X_irpt[i, p, r, t].x

    return solution, constraint_times