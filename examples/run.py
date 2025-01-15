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
            pl.lit(100.)
        )
        .otherwise(
            pl.lit(0.)
        )
        .alias("low"),
        pl
        .when(
            pl.col("category") == "a",
        )
        .then(
            pl.lit(1_000.)
        )
        .otherwise(
            pl.lit(1.)
        )
        .alias("high"),

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

        pl
        .when(
            pl.col("category") == "a",
        )
        .then(
            pl.lit(100)
        )
        .otherwise(
            pl.lit(10)
        )
        .cast(pl.UInt64)
        .alias("n"),
        pl
        .when(
            pl.col("category") == "a",
        )
        .then(
            pl.lit(.0001)
        )
        .otherwise(
            pl.lit(.99)
        )
        .alias("p"),
    )
)

print(
    df
    .random.rand(seed=42)
    .random.rand(low=pl.col("low"), high=pl.col("high"), seed=42)
    .random.normal(seed=42, name="normal_seed_1")
    .random.normal(seed=42, name="normal_seed_2")
    .random.normal(mean="mean", std="std", seed=42, name="normal_expr")
    .random.normal(mean=pl.col("mean"), std=pl.col("std"), seed=42, name="normal_expr_2")
    .random.binomial(n=24, p=.5, seed=42)
    .random.binomial(n="n", p="p", name="binomial_expr")
    .random.binomial(n=pl.col("n"), p=pl.col("p"), name="binomial_expr_2")
)