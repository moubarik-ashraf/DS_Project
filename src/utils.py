# ============================================================
# Bi-objective scheduling with epsilon-constraint (PuLP)
# 4 jobs, 2 stages, machines per stage = (2, 2)
# Criteria: C_max (makespan) and total cost
# ============================================================

import pulp as pl
import numpy as np
import pandas as pd
import plotly.express as px

# -----------------------------
# Sets and data
# -----------------------------
J = range(4)          # jobs: 0..3
E = range(2)          # stages: 0..1
machines_per_stage = {0: 2, 1: 2}  # 2 machines à chaque étage
M = {e: range(machines_per_stage[e]) for e in E}
K = range(4)          # positions 0..3 (une par job)

# Processing times p[j,e,l] (toy mais déterministe)
p = {
    (j, e, l): (j + 1) + 2*(e + 1) + (l + 1)
    for j in J for e in E for l in M[e]
}

# Costs c[j,e,l] (toy data)
c = {
    (j, e, l): 10*(l + 1) + 3*(e + 1) + (j % 3)
    for j in J for e in E for l in M[e]
}

# Big-M réduit (adapté à la taille des p)
BIGM = 200


# -----------------------------
# Model builder with epsilon-constraint
# objective_type: "cmax" or "cost"
# epsilon: None for pure single-objective, or value for C_cost <= epsilon
# -----------------------------
def solve_model(epsilon=None, objective_type="cmax"):
    prob = pl.LpProblem("ParallelFlowshop_Epsilon", pl.LpMinimize)

    # Decision variables
    X = {(j, e, k): pl.LpVariable(f"X_{j}_{e}_{k}", 0, 1, cat="Binary")
         for j in J for e in E for k in K}
    Y = {(e, k, l): pl.LpVariable(f"Y_{e}_{k}_{l}", 0, 1, cat="Binary")
         for e in E for k in K for l in M[e]}
    Z = {(j, e, k, l): pl.LpVariable(f"Z_{j}_{e}_{k}_{l}", 0, 1, cat="Binary")
         for j in J for e in E for k in K for l in M[e]}

    C = {(e, k): pl.LpVariable(f"C_{e}_{k}", lowBound=0)
         for e in E for k in K}
    Cmax = pl.LpVariable("Cmax", lowBound=0)
    Ccost = pl.LpVariable("Ccost", lowBound=0)

    # -----------------------------
    # Objective
    # -----------------------------
    if objective_type == "cmax":
        prob += Cmax
    elif objective_type == "cost":
        prob += Ccost
    else:
        raise ValueError("objective_type must be 'cmax' or 'cost'")

    # -----------------------------
    # Constraints
    # -----------------------------

    # Makespan: Cmax >= completion time à dernier étage
    last_stage = max(E)
    for k in K:
        prob += Cmax >= C[(last_stage, k)]

    # 1) Chaque position k à l'étage e a exactement un job
    for e in E:
        for k in K:
            prob += pl.lpSum(X[(j, e, k)] for j in J) == 1

    # 2) Chaque job j apparaît exactement une fois à l'étage e
    for e in E:
        for j in J:
            prob += pl.lpSum(X[(j, e, k)] for k in K) == 1

    # 3) Chaque position (e,k) est traitée sur exactement une machine
    for e in E:
        for k in K:
            prob += pl.lpSum(Y[(e, k, l)] for l in M[e]) == 1

    # 4) Temps de fin à l'étage 0
    e0 = 0
    for k in K:
        for l in M[e0]:
            prob += C[(e0, k)] >= pl.lpSum(X[(j, e0, k)] * p[(j, e0, l)] for j in J) \
                                 - BIGM * (1 - Y[(e0, k, l)])

    # 5) Ordonnancement sur une même machine au même étage
    for e in E:
        for l in M[e]:
            for k in K:
                for q in K:
                    if k > q:
                        prob += C[(e, k)] >= C[(e, q)] \
                            + pl.lpSum(X[(j, e, k)] * p[(j, e, l)] for j in J) \
                            - BIGM * (2 - Y[(e, k, l)] - Y[(e, q, l)])

    # 6) Précédence entre étages consécutifs (même job)
    # ici il n'y a que e=1 qui suit e=0
    for e in E:
        if e == 0:
            continue
        for l in M[e]:
            for j in J:
                for k in K:
                    for q in K:
                        prob += C[(e, k)] >= C[(e - 1, q)] + p[(j, e, l)] \
                            - BIGM * (3 - Y[(e, k, l)] - X[(j, e, k)] - X[(j, e - 1, q)])

    # 7) Linéarisation Z = X * Y
    for j in J:
        for e in E:
            for k in K:
                for l in M[e]:
                    prob += Z[(j, e, k, l)] <= X[(j, e, k)]
                    prob += Z[(j, e, k, l)] <= Y[(e, k, l)]
                    prob += Z[(j, e, k, l)] >= X[(j, e, k)] + Y[(e, k, l)] - 1

    # 8) Définition du coût total
    prob += Ccost == pl.lpSum(
        c[(j, e, l)] * Z[(j, e, k, l)]
        for e in E for k in K for j in J for l in M[e]
    )

    # 9) Contrainte epsilon sur le coût (si utilisée)
    if epsilon is not None:
        prob += Ccost <= epsilon

    # -----------------------------
    # Solve
    # -----------------------------
    status = prob.solve(pl.PULP_CBC_CMD(msg=False))
    status_str = pl.LpStatus[status]

    if status_str != "Optimal":
        return None, None, status_str

    Cmax_val = pl.value(Cmax)
    Ccost_val = pl.value(Ccost)
    return Cmax_val, Ccost_val, status_str


# ============================================================
# Step 1: get min cost (objective = cost)
# ============================================================
cmax_cost_opt, cost_min, status1 = solve_model(objective_type="cost")
print("Min cost solution status:", status1, "| Cmax:", cmax_cost_opt, "| Cost_min:", cost_min)

# ============================================================
# Step 2: epsilon-grid and epsilon-constraint runs
# ============================================================

# 6 valeurs d'epsilon entre cost_min et 2 * cost_min
num_eps = 6
epsilons = np.linspace(cost_min, 2.0 * cost_min, num_eps)

results = []
for eps in epsilons:
    cmax_val, cost_val, status = solve_model(epsilon=eps, objective_type="cmax")
    print(f"epsilon={eps:.1f} -> status={status}, Cmax={cmax_val}, Cost={cost_val}")
    if status == "Optimal":
        results.append({"epsilon": eps, "Cmax": cmax_val, "Cost": cost_val})

df = pd.DataFrame(results).drop_duplicates(subset=["Cmax", "Cost"])

# ============================================================
# Step 3: compute Pareto frontier (non-dominated points)
# Minimiser les deux: Cost et Cmax
# ============================================================

def pareto_front(df, cost_col="Cost", cmax_col="Cmax"):
    df_sorted = df.sort_values(by=[cost_col, cmax_col]).reset_index(drop=True)
    pareto = []
    best_cmax = float("inf")
    for _, row in df_sorted.iterrows():
        if row[cmax_col] < best_cmax:
            pareto.append(row)
            best_cmax = row[cmax_col]
    return pd.DataFrame(pareto)

df_pareto = pareto_front(df)

print("\nPareto points:")
print(df_pareto)

# ============================================================
# Step 4: Plot Pareto frontier with Plotly
# ============================================================

fig = px.scatter(
    df,
    x="Cost",
    y="Cmax",
    color="epsilon",
    title="Solutions (epsilon-constraint) - 4 jobs, 2 stages, 2 machines/stage",
    labels={"Cost": "Total Cost", "Cmax": "Makespan"}
)
fig.update_traces(mode="markers")

fig_pareto = px.line(
    df_pareto,
    x="Cost",
    y="Cmax"
)
fig_pareto.update_traces(line=dict(dash="dash"), showlegend=False)

for trace in fig_pareto.data:
    fig.add_trace(trace)

fig.show()
