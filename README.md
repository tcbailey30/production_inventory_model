# IMSE 780 — Production and Inventory Optimization (Baseline V1.0)

This package contains a **complete, reproducible Python integer programming model** for the fictional products P1–P4 across the four quarters of FY2027.

## Contents

| File | Purpose |
| --- | --- |
| `IMSE780_Production_Inventory_Model.py` | Documented optimization model and verification checks |
| `IMSE780_FY2027_Model_Inputs.csv` | Sixteen product-quarter input rows extracted from the clean V1.6 dataset |
| `production_inventory_schedule.csv` | Verified optimal decisions and inventory balances |
| `quarterly_budget_verification.csv` | Verified quarterly spending versus available budgets |
| `model_summary.txt` | Solver result and important totals |

The original source Excel workbook is `IMSE780_MILP_Clean_Dataset_V1_6.xlsx` (supplied separately). The model runs directly from the CSV snapshot to keep its dependencies simple. **If you revise Excel inputs, also update the CSV before running the model again.**

## How to run

1. Install Python 3.10 or later.
2. Open Terminal (macOS) or Command Prompt (Windows) in the folder containing the `.py` and `.csv` files.
3. Install packages:

   ```bash
   python3 -m pip install numpy scipy
   ```

4. Solve the model:

   ```bash
   python3 IMSE780_Production_Inventory_Model.py
   ```

   On Windows, use `python` or `py` instead of `python3` if appropriate.

5. Look in the automatically created `IMSE780_Model_Results/` folder for the CSV results and summary.

Alternatively, specify paths:

```bash
python3 IMSE780_Production_Inventory_Model.py --inputs IMSE780_FY2027_Model_Inputs.csv --output output
```

## Mathematical formulation

**Sets:** \(I=\{P1,P2,P3,P4\}\), \(T=\{FY\ Q1, FY\ Q2, FY\ Q3, FY\ Q4\}\).

**Decision variables:** \(x_{it}\) = units produced; \(S_{it}\) = end-of-quarter inventory; both nonnegative integers.

**Parameters:** \(c_{it}\) = production cost, \(h_{it}\) = inventory holding cost, \(a_{it}\) = scheduled receipts, \(D_{it}\) = required total demand, \(R_{it}\) = reserve floor, \(P_{it}^{max}\) = production capacity, and \(B_t\) = quarterly budget.

**Objective:**

\[
\min Z=\sum_{i\in I}\sum_{t\in T}(c_{it}x_{it}+h_{it}S_{it})
\]

**Constraints:**

\[
\begin{aligned}
S_{it} &= S_{i,t-1}+x_{it}+a_{it}-D_{it} &&\forall i,t\\
S_{it} &\ge R_{it} &&\forall i,t\\
x_{it} &\le P_{it}^{max} &&\forall i,t\\
\sum_{i\in I}(c_{it}x_{it}+h_{it}S_{it}) &\le B_t &&\forall t\\
S_{i0} &= I_i^0 &&\forall i\\
x_{it},S_{it} &\in\mathbb{Z}_{\ge0} &&\forall i,t.
\end{aligned}
\]

**Assumptions:** Demand is deterministic and must be fully met (mandatory + flexible are both included). Fixed scheduled receipts become available within their designated quarter. Newly produced units also become available in the same quarter. Reserve floors are protected ending inventory rather than extra consumed demand. Inventory holding costs are charged against ending inventory. No setup costs, variable labor costs, unmet-demand penalties, or real-world missile stockpile or production data are included.

**Terminology:** As written, this is a pure **integer linear program (ILP)** rather than a strictly mixed-integer linear program (MILP), because both classes of decision variables are integer. It is solved with the HiGHS mixed-integer solver through `scipy.optimize.milp`.

## Expected results with the provided input CSV

- **Optimal objective:** 1331.95 fictional cost units (CU)
- Production portion: 1258.00 CU
- Holding portion: 73.95 CU
- Production units: 181
- Demand satisfied: 376
- Sum of end-of-FY2027 inventories: 35
- All 16 inventory balances, reserves, production ceilings, and all four quarterly budgets pass the script's independent checks.

**Data provenance:** The values are independently invented for this academic simulation. Public missile prices and appropriations in the V1.6 workbook are research context only and **not coefficients** in this optimization.
