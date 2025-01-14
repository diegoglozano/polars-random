import polars as pl
import polars_random

df = pl.DataFrame(
    {
        "a": [0.2, 0.5, 0.7, 0.8, 0.9],
    }
)

print(
    df
    .random.rand(seed=42)
    .random.normal(seed=42, name="normal_seed_1")
    .random.normal(seed=42, name="normal_seed_2")
    .random.binomial(24, .5, seed=42)
)