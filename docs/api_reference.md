# API reference

`polars-random` exposes the same set of distributions through four interchangeable entry points. Pick whichever fits your pipeline; the underlying Rust kernel is the same.

## Top-level functions

Returns a `pl.Expr` by default, or a `pl.Series` of length `size` when `size=` is given.

::: polars_random.rand
    handler: python

::: polars_random.uniform
    handler: python

::: polars_random.normal
    handler: python

::: polars_random.binomial
    handler: python

::: polars_random.randint
    handler: python

## Global seed

Set one seed for the whole session. Any draw that omits `seed=` then derives its
seed from this global generator; an explicit `seed=` on a call still overrides it.
Distinct expressions consume the generator separately, so they stay independent
while remaining reproducible across re-runs.

```python
import polars as pl
import polars_random as pr

pr.set_random_seed(42)
df = pl.DataFrame({"id": range(5)})
df.with_columns(a=pr.normal(), b=pr.rand())  # reproducible, no per-call seed
```

Reproducibility depends on the order and number of seedless draws (each takes
the next value from the generator, like NumPy's or Polars' global RNG). To make
two columns *identical*, give them the same explicit `seed=` — the global seed
is designed to keep seedless draws independent:

```python
df.with_columns(a=pr.normal(seed=7), b=pr.normal(seed=7))  # a == b
```

`pr.set_random_seed` is independent of `polars.set_random_seed` (which seeds
Polars' own `.sample()` / `.shuffle()` and is not readable by plugins).

::: polars_random.set_random_seed
    handler: python

## `pl.col(...).random` — expression namespace

Use inside any expression context (`select`, `with_columns`, lazy queries, group-by aggregations, …). The parent expression provides the row count.

```python
import polars as pl
import polars_random  # registers the namespace

df.with_columns(noise=pl.col("id").random.normal(mean=0, std=1, seed=42))
```

Available methods: `rand` / `uniform`, `normal`, `binomial`, `randint`. Same parameters as the top-level functions, minus `size`.

## `df.random` — DataFrame namespace

::: polars_random.Random
    handler: python

## `lf.random` — LazyFrame namespace

Same API as `df.random` but returns a `pl.LazyFrame`. Lets the random draws stay inside a lazy plan and be optimized alongside the rest of your query.

```python
(
    df.lazy()
      .filter(pl.col("active"))
      .random.normal(seed=42, name="noise")
      .collect()
)
```

## Null handling

When a parameter is supplied as a column or expression, any null in that column is propagated to the output as `null` instead of raising. Scalar parameters are validated up front (`seed >= 0`, `0 <= p <= 1`, valid distribution params) and raise `ValueError` / `PolarsError` if invalid.
