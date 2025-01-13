import polars as pl
import polars_random
import os

os.environ["POLARS_VERBOSE"] = "1"

df = pl.DataFrame(
    {
        "a": [0.2, 0.5, 0.7, 0.8, 0.9],
    }
).with_columns(
    pl.col("a").random.rand().alias("rand"),
    pl.col("a").random.rand(low=10, high=11, seed=42).alias("rand_10_11"),
    pl.col("a").random.normal().alias("normal"),
    pl.col("a").random.normal(seed=42).alias("normal_seed1"),
    pl.col("a").random.normal(seed=42).alias("normal_seed2"),
    pl.col("a").random.binomial(n=10, p=.3).alias("binomial"),
)
print(df)