from pathlib import Path

import numpy
import pandas


HERE = Path(__file__).resolve().parent
ROWS = 600
NEW_ROWS = 20
PLANS = ("basic", "plus", "pro")
CONTRACTS = ("monthly", "yearly", "two_year")


def churn_frame(rows=ROWS, seed=0):
    generator = numpy.random.default_rng(seed)
    tenure = generator.integers(1, 72, rows).astype("float64")
    monthly = generator.normal(60.0, 20.0, rows).clip(10.0, None).round(2)
    calls = generator.poisson(1.5, rows).astype("float64")
    plan = generator.choice(PLANS, rows, p=[0.5, 0.3, 0.2])
    contract = generator.choice(CONTRACTS, rows, p=[0.55, 0.3, 0.15])
    score = (-1.2 - 0.04 * tenure + 0.025 * (monthly - 60.0) + 0.6 * calls
             + numpy.select([contract == "monthly", contract == "yearly"], [1.0, -0.3], -1.2)
             + numpy.select([plan == "basic", plan == "plus"], [0.3, 0.0], -0.4))
    churned = generator.random(rows) < 1.0 / (1.0 + numpy.exp(-score))
    return pandas.DataFrame({"num_tenure": tenure, "num_monthly_fee": monthly, "num_support_calls": calls,
                             "cat_plan": plan, "cat_contract": contract,
                             "churned": numpy.where(churned, "yes", "no")})


def main():
    path = HERE / "churn.parquet"
    frame = churn_frame()
    frame.to_parquet(path, index=False)
    new = HERE / "new.parquet"
    churn_frame(NEW_ROWS, seed=1).drop(columns=["churned"]).to_parquet(new, index=False)
    share = float((frame["churned"] == "yes").mean())
    print(f"wrote {path} ({ROWS} rows, {share:.0%} churned) and {new} ({NEW_ROWS} customers without churned)")


if __name__ == "__main__":
    main()
