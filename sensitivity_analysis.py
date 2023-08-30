# Constraint 9
'''start_time = time.time()
for i in range(1, I):
    lhs10 = gp.quicksum(X_irpt[i, p, r, t] * W[p][r - 1] * Q[i] for p in range(1, P + 1) for r in range(1, R + 1) for t in range(1, T+1))
    model.addConstr(lhs10 <= max_d)
    model.addConstr(lhs10 >= min_d)
end_time = time.time()
constraint_times["Constraint 9"] = end_time - start_time'''

# Constraint 6
'''start_time = time.time()
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
constraint_times["Constraint 6"] = end_time - start_time'''

