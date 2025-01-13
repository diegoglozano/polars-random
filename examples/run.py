import polars as pl
import polars_random

df = pl.DataFrame(
    {
        "a": [0.2, 0.5, 0.7, 0.8, 0.9],
    }
).with_columns(
    pl.col("a").random.rand().alias("rand"),
    pl.col("a").random.normal().alias("normal"),
    pl.col("a").random.normal(seed=42).alias("normal_seed1"),
)
print(df)