import random

def solve_crop_optimization_heuristic2(I, R, P, T, H, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples, max_iterations):
    # Initialize the solution with all decision variables set to 0
    solution = {f"X_{i}_{p}_{r}_{t}": 0 for i in range(1, I + 1) for p in range(1, P + 1)
                for r in range(1, R + 1) for t in range(1, T + 1)}

    # Randomly initialize the solution
    initialize_random_solution(solution, I, R, P, T)

    # Iterate until all requirements are met or maximum iterations reached
    iteration = 0
    while not are_requirements_met(solution, I, R, P, T, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples) and iteration < max_iterations:
        # Select a random decision variable to update
        i, p, r, t = select_random_variable(I, R, P, T)

        # Explore the neighborhood by checking other possible values for the selected variable
        best_solution = solution.copy()
        best_objective = calculate_objective(best_solution, W)

        for value in range(2):  # Try both 0 and 1
            solution[f"X_{i}_{p}_{r}_{t}"] = value
            if are_requirements_met(solution, I, R, P, T, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples):
                objective = calculate_objective(solution, W)
                if objective < best_objective:
                    best_solution = solution.copy()
                    best_objective = objective

        # Update the solution with the best found in the neighborhood
        solution = best_solution

        iteration += 1

    return solution


def initialize_random_solution(solution, I, R, P, T):
    for i in range(1, I + 1):
        for p in range(1, P + 1):
            for r in range(1, R + 1):
                for t in range(1, T + 1):
                    solution[f"X_{i}_{p}_{r}_{t}"] = random.randint(0, 1)


def are_requirements_met(solution, I, R, P, T, theta, W, A, Q, C, S, Z, G, F, adjacent_tuples):
    # Constraint 1
    for i in range(1, I + 1):
        for t in range(1, T + 1):
            lhs = sum(solution[f"X_{i}_{p}_{r}_{t}"] * W[p][r - 1] for p in range(1, P + 1) for r in range(1, R + 1))
            rhs = sum(W[p][r - 1] for p in range(1, P + 1) for r in range(1, R + 1))
            if lhs > rhs:
                return False

    # Constraint 2
    for i in range(1, I + 1):
        lhs_3 = sum(solution[f"X_{i}_{p}_{r}_{t}"] * W[p][r - 1] * Q[i]
                    for p in range(1, P + 1) for r in range(1, R + 1) for t in range(1, T + 1))
        if lhs_3 > 40:
            return False

    # Constraint 3
    for t in range(1, T + 1):
        for p in range(1, P + 1):
            for r in range(1, R + 1):
                lhs_1 = sum(
                    solution[f"X_{i}_{p}_{r}_{t - z + T}"] if (t - z) <= 0 else solution[f"X_{i}_{p}_{r}_{t - z}"]
                    for i in range(1, I + 1)
                    for z in range(0, theta[i])
                )
                if lhs_1 > 1:
                    return False

    # Constraint 4
    for r in range(1, R + 1):
        for p in range(1, P + 1):
            lhs3 = sum(solution[f"X_{I}_{p}_{r}_{t}"] for t in range(1, T + 1))
            if lhs3 != 1:
                return False

    # Constraint 5
    for t in range(1, T + 1):
        for p in range(1, P + 1):
            lhs4 = sum(solution[f"X_{i}_{p}_{r}_{t}"] for i in C[1] for r in S[2]) * sum(solution[f"X_{i}_{p}_{r}_{t}"] for i in C[1] for r in S[3])
            lhs5 = sum(solution[f"X_{i}_{p}_{r}_{t}"] for i in C[2] for r in S[3])
            lhs6 = sum(solution[f"X_{i}_{p}_{r}_{t}"] for i in C[3] for r in S[1])
            if lhs4 > 0 or lhs5 > 0 or lhs6 > 0:
                return False

    # Constraint 6
    for i in range(1, I + 1):
        for t in range(1, T + 1):
            for p in range(1, P + 1):
                for r in range(1, R + 1):
                    if solution[f"X_{i}_{p}_{r}_{t}"] == 1 and Z[i] * solution[f"X_{i}_{p}_{r}_{t}"] > G[p, r]:
                        return False

    # Constraint 7
    for h in F:
        for t in range(1, T + 1):
            for (p1, r1, p2, r2) in adjacent_tuples:
                lhs7 = sum(
                    solution[f"X_{i}_{p1}_{r1}_{t - z + T}"] + solution[f"X_{i}_{p2}_{r2}_{t - z + T}"] if (t - z) <= 0 else solution[f"X_{i}_{p1}_{r1}_{t - z}"] +
                                                                                                     solution[f"X_{i}_{p2}_{r2}_{t - z}"]
                    for i in F[h]
                    for z in range(0, theta[i]))
                if lhs7 > 1:
                    return False

    return True


def calculate_objective(solution, W, A, Q):
    objective = sum(solution[f"X_{i}_{p}_{r}_{t}"] * W[p][r - 1] * A[i][t] * Q[i]
                    for i, p, r, t in solution.keys())

    return objective

def select_random_variable(I, R, P, T):
    i = random.randint(1, I)
    p = random.randint(1, P)
    r = random.randint(1, R)
    t = random.randint(1, T)
    return i, p, r, t
