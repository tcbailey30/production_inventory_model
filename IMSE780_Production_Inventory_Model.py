"""
IMSE 780 — Deterministic Multi-Period Production and Inventory Model
--------------------------------------------------------------------
Fictional products P1–P4, federal fiscal quarters FY Q1–FY Q4.

Objective: minimize production costs + ending-inventory holding costs.
Constraints:
  (1) Inventory balance, including fixed receipts and required demand.
  (2) Minimum ending-inventory reserve.
  (3) Product-by-quarter production capacity.
  (4) Quarter-wide spending budget.
  (5) Production and inventory are nonnegative INTEGERS.

This is a pure Integer Linear Program (ILP). It can be solved with a
mixed-integer optimization solver, but it is not strictly a MILP until a
meaningful continuous decision variable is added.

All quantities/costs are fictional classroom inputs. No setup-cost,
labor-cost, shortages, or real-world munitions data are modeled.

Usage:
    python -m pip install numpy scipy
    python IMSE780_Production_Inventory_Model.py
    python IMSE780_Production_Inventory_Model.py --inputs my_inputs.csv --output results

Requires IMSE780_FY2027_Model_Inputs.csv alongside this script by default.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


QUARTERS = ("FY Q1", "FY Q2", "FY Q3", "FY Q4")
INTEGER_FIELDS = (
    "opening_inventory", "demand", "mandatory_demand", "flexible_demand",
    "scheduled_receipts", "reserve_floor", "production_capacity",
)
COST_FIELDS = ("production_cost_CU", "holding_cost_CU", "quarterly_budget_CU")


def load_inputs(path: Path):
    """Read the workbook-derived input table and check basic consistency."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        expected = {"product", "quarter", *INTEGER_FIELDS, *COST_FIELDS}
        missing = expected - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        records = list(reader)

    inputs = {}
    for row in records:
        p = row["product"].strip()
        t = row["quarter"].strip()
        if not p or t not in QUARTERS:
            raise ValueError(f"Invalid product or fiscal quarter: {p!r}, {t!r}")
        if (p, t) in inputs:
            raise ValueError(f"Duplicate product-quarter combination: {p}, {t}")
        item = {k: int(row[k]) for k in INTEGER_FIELDS}
        item.update({k: float(row[k]) for k in COST_FIELDS})
        if any(val < 0 for val in item.values()):
            raise ValueError(f"Negative input for {p}, {t}")
        if item["mandatory_demand"] + item["flexible_demand"] != item["demand"]:
            raise ValueError(f"Demand component mismatch for {p}, {t}")
        inputs[p, t] = item

    products = sorted({p for p, _ in inputs})
    expected_keys = {(p, t) for p in products for t in QUARTERS}
    if set(inputs) != expected_keys:
        raise ValueError(f"Each product must have all four quarters: {expected_keys - set(inputs)}")
    for p in products:
        openings = {inputs[p, t]["opening_inventory"] for t in QUARTERS}
        if len(openings) != 1:
            raise ValueError(f"Inconsistent opening inventory for {p}")
    for t in QUARTERS:
        budgets = {inputs[p, t]["quarterly_budget_CU"] for p in products}
        if len(budgets) != 1:
            raise ValueError(f"Inconsistent shared quarterly budget for {t}")
    return products, inputs


def build_and_solve(products, data):
    """Construct the ILP using SciPy's HiGHS MILP interface."""
    pairs = [(p, t) for p in products for t in QUARTERS]
    count = len(pairs)

    # Decision-vector ordering: [x(P1,Q1), ... x(P4,Q4), S(P1,Q1), ... S(P4,Q4)]
    x_index = {pair: j for j, pair in enumerate(pairs)}
    s_index = {pair: count + j for j, pair in enumerate(pairs)}
    variable_count = 2 * count

    # Objective: min sum_{i,t} (c_it*x_it + h_it*S_it)
    objective = np.zeros(variable_count)
    for pair in pairs:
        objective[x_index[pair]] = data[pair]["production_cost_CU"]
        objective[s_index[pair]] = data[pair]["holding_cost_CU"]

    rows, lower, upper = [], [], []

    def add_constraint(coefficients, minimum=-np.inf, maximum=np.inf):
        row = np.zeros(variable_count)
        for variable_index, coefficient in coefficients.items():
            row[variable_index] = coefficient
        rows.append(row)
        lower.append(minimum)
        upper.append(maximum)

    # CONSTRAINT 1: S_it = S_i,t-1 + x_it + receipts_it - demand_it
    # Rearranged: S_it - x_it - S_i,t-1 = receipts_it - demand_it.
    # For Q1, replace previous inventory with the fixed opening inventory.
    for p, t in pairs:
        item = data[p, t]
        coeffs = {s_index[p, t]: 1.0, x_index[p, t]: -1.0}
        rhs = item["scheduled_receipts"] - item["demand"]
        qidx = QUARTERS.index(t)
        if qidx == 0:
            rhs += item["opening_inventory"]
        else:
            previous = QUARTERS[qidx - 1]
            coeffs[s_index[p, previous]] = -1.0
        add_constraint(coeffs, minimum=rhs, maximum=rhs)

    # CONSTRAINT 2: S_it >= reserve_floor_it
    for pair in pairs:
        add_constraint({s_index[pair]: 1.0}, minimum=data[pair]["reserve_floor"])

    # CONSTRAINT 3: x_it <= production_capacity_it
    for pair in pairs:
        add_constraint({x_index[pair]: 1.0}, maximum=data[pair]["production_capacity"])

    # CONSTRAINT 4: sum_i (production_cost_it*x_it + holding_cost_it*S_it) <= budget_t
    for t in QUARTERS:
        coeffs = {}
        for p in products:
            item = data[p, t]
            coeffs[x_index[p, t]] = item["production_cost_CU"]
            coeffs[s_index[p, t]] = item["holding_cost_CU"]
        budget = data[products[0], t]["quarterly_budget_CU"]
        add_constraint(coeffs, maximum=budget)

    # CONSTRAINT 5: x_it, S_it are nonnegative integers.
    # integrality=1 requires integer values; lower bounds enforce nonnegativity.
    constraints = LinearConstraint(np.vstack(rows), np.array(lower), np.array(upper))
    bounds = Bounds(np.zeros(variable_count), np.full(variable_count, np.inf))
    integrality = np.ones(variable_count, dtype=int)

    result = milp(c=objective, integrality=integrality,
                  bounds=bounds, constraints=constraints,
                  options={"disp": False, "mip_rel_gap": 0.0})
    if not result.success or result.x is None:
        raise RuntimeError(f"No proven optimal solution: {result.message}. "
                           "Check demand, reserves, production limits, and budgets.")

    # Values are integers in theory; round within tight numerical tolerance.
    if np.max(np.abs(result.x - np.rint(result.x))) > 1e-6:
        raise RuntimeError("Solver returned a non-integer solution outside tolerance")
    solution = np.rint(result.x).astype(int)
    x = {pair: int(solution[x_index[pair]]) for pair in pairs}
    s = {pair: int(solution[s_index[pair]]) for pair in pairs}

    # Independent verification of every modeled constraint.
    per_product_rows = []
    per_quarter_cost = {t: 0.0 for t in QUARTERS}
    for p in products:
        begin = data[p, QUARTERS[0]]["opening_inventory"]
        for t in QUARTERS:
            d = data[p, t]
            produced, ending = x[p, t], s[p, t]
            predicted = begin + produced + d["scheduled_receipts"] - d["demand"]
            assert ending == predicted, f"Inventory balance error: {p}, {t}"
            assert ending >= d["reserve_floor"], f"Reserve violation: {p}, {t}"
            assert 0 <= produced <= d["production_capacity"], f"Capacity violation: {p}, {t}"
            cost_production = d["production_cost_CU"] * produced
            cost_holding = d["holding_cost_CU"] * ending
            cost_total = cost_production + cost_holding
            per_quarter_cost[t] += cost_total
            per_product_rows.append({
                "product": p, "quarter": t, "beginning_inventory": begin,
                "production": produced, "scheduled_receipts": d["scheduled_receipts"],
                "demand_satisfied": d["demand"], "ending_inventory": ending,
                "reserve_floor": d["reserve_floor"],
                "production_capacity": d["production_capacity"],
                "production_cost_CU": round(cost_production, 4),
                "holding_cost_CU": round(cost_holding, 4),
                "total_cost_CU": round(cost_total, 4),
            })
            begin = ending

    budget_rows = []
    for t in QUARTERS:
        budget = data[products[0], t]["quarterly_budget_CU"]
        spend = per_quarter_cost[t]
        assert spend <= budget + 1e-6, f"Quarterly budget violation: {t}"
        budget_rows.append({
            "quarter": t, "spending_CU": round(spend, 4),
            "budget_CU": round(budget, 4),
            "unused_budget_CU": round(budget - spend, 4),
        })

    total_from_rows = sum(r["total_cost_CU"] for r in per_product_rows)
    assert abs(total_from_rows - result.fun) <= 1e-5, "Objective mismatch"
    return result, per_product_rows, budget_rows


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    home = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path,
                        default=home / "IMSE780_FY2027_Model_Inputs.csv")
    parser.add_argument("--output", type=Path,
                        default=home / "IMSE780_Model_Results")
    args = parser.parse_args()

    products, data = load_inputs(args.inputs)
    result, schedule, quarterly = build_and_solve(products, data)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "production_inventory_schedule.csv", schedule)
    write_csv(args.output / "quarterly_budget_verification.csv", quarterly)

    total_produced = sum(row["production"] for row in schedule)
    total_demand = sum(row["demand_satisfied"] for row in schedule)
    final_inventory = sum(row["ending_inventory"] for row in schedule
                          if row["quarter"] == QUARTERS[-1])
    production_cost = sum(row["production_cost_CU"] for row in schedule)
    holding_cost = sum(row["holding_cost_CU"] for row in schedule)
    report = [
        "IMSE 780 — BASELINE MODEL RESULTS (synthetic classroom data)",
        "Solver: SciPy scipy.optimize.milp (HiGHS)",
        "Model: pure ILP; x_it and S_it are nonnegative integers",
        f"Solver status: {result.message}",
        f"Minimum total cost: {result.fun:.2f} CU",
        f"  Production cost: {production_cost:.2f} CU",
        f"  Holding cost: {holding_cost:.2f} CU",
        f"Total units produced: {total_produced}",
        f"Total demand satisfied: {total_demand}",
        f"Final total inventory: {final_inventory}",
        "Checked: every inventory balance, reserve, capacity, budget, and integrality",
    ]
    (args.output / "model_summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nDetailed results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
