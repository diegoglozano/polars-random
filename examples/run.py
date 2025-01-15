import polars as pl
import polars_random as plr

df = (
    pl
    .DataFrame(
        {
            "category": ["a", "b", "a", "a", "b"],
        }
    )
    .with_columns(
        pl
        .when(
            pl.col("category") == "a",
        )
        .then(
            pl.lit(2.)
        )
        .otherwise(
            pl.lit(0.)
        )
        .alias("mean"),
        pl
        .when(
            pl.col("category") == "a",
        )
        .then(
            pl.lit(.4)
        )
        .otherwise(
            pl.lit(1.)
        )
        .alias("std"),
    )
)

print(
    df
    .random.rand(seed=42)
    .random.normal(seed=42, name="normal_seed_1")
    .random.normal(seed=42, name="normal_seed_2")
    .random.normal(mean="mean", std="std", seed=42, name="normal_expr")
    .random.normal(mean=pl.col("mean"), std=pl.col("std"), seed=42, name="normal_expr_2")
    .random.binomial(24, .5, seed=42)
)