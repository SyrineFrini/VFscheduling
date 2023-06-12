import random

def solve_crop_optimization_heuristic(I, R, P, T, H, theta, W, A, Q, C, S, Z, G, F, max_iterations=50):
    # Initialize the solution with all decision variables set to 0
    solution = {f"X_{i}_{p}_{r}_{t}": 0 for i in range(1, I + 1) for p in range(1, P + 1)
                for r in range(1, R + 1) for t in range(1, T + 1)}

    # Iterate until all requirements are met or maximum iterations reached
    iteration = 0
    while not are_requirements_met(solution, I, R, P, T, theta, W, A, Q, C, S, Z, G, F) and iteration < max_iterations:
        # Randomly select a decision variable to update
        i, p, r, t = select_random_variable(I, R, P, T)
        solution[f"X_{i}_{p}_{r}_{t}"] = 1  # Set the selected variable to 1

        iteration += 1

    return solution


def are_requirements_met(solution, I, R, P, T, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples):
    # Constraint 1: Maximum capacity of each shelf
    for r in range(1, R + 1):
        for p in range(1, P + 1):
            shelf_capacity = G[p, r]
            used_capacity = sum(solution[f"X_{i}_{p}_{r}_{t}"] * W[p][r - 1]
                                for i in range(1, I + 1) for t in range(1, T + 1))
            if used_capacity > shelf_capacity:
                return False

    # Constraint 2: Limit the maximum number of crops per period
    for i in range(1, I + 1):
        for t in range(1, T + 1):
            crop_count = sum(solution[f"X_{i}_{p}_{r}_{t}"] for p in range(1, P + 1) for r in range(1, R + 1))
            if crop_count > theta[i]:
                return False

    # Constraint 3: Maintenance period for each shelf
    for r in range(1, R + 1):
        for p in range(1, P + 1):
            maintenance_period = sum(solution[f"X_{19}_{p}_{r}_{t}"] for t in range(1, T + 1))
            if maintenance_period != 1:
                return False

    # Constraint 4: Sun categories
    for t in range(1, T + 1):
        for p in range(1, P + 1):
            if any(solution[f"X_{i}_{p}_{r}_{t}"] == 1 for i in C[1] for r in S[2]):
                if any(solution[f"X_{i}_{p}_{r}_{t}"] == 1 for i in C[1] for r in S[3]):
                    return False
            if any(solution[f"X_{i}_{p}_{r}_{t}"] == 1 for i in C[2] for r in S[3]):
                return False
            if any(solution[f"X_{i}_{p}_{r}_{t}"] == 1 for i in C[3] for r in S[1]):
                return False

    # Constraint 5: Shelves height
    for i in range(1, I + 1):
        for t in range(1, T + 1):
            for p in range(1, P + 1):
                for r in range(1, R + 1):
                    if solution[f"X_{i}_{p}_{r}_{t}"] == 1 and Z[i] * solution[f"X_{i}_{p}_{r}_{t}"] > G[p, r]:
                        return False

    # Constraint 6: Adjacent shelves
    for h in F:
        for t in range(1, T + 1):
            for (p1, r1, p2, r2) in adjacent_tuples:
                if any(solution[f"X_{i}_{p1}_{r1}_{t - z + T}"] + solution[f"X_{i}_{p2}_{r2}_{t - z + T}"]
                       if (t - z) <= 0 else solution[f"X_{i}_{p1}_{r1}_{t - z}"] + solution[f"X_{i}_{p2}_{r2}_{t - z}"]
                       for i in F[h] for z in range(0, theta[i])):
                    return False

    return True

def select_random_variable(I, R, P, T):
    i = random.randint(1, I)
    p = random.randint(1, P)
    r = random.randint(1, R)
    t = random.randint(1, T)
    return i, p, r, t

