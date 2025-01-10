from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

from polars_random._internal import __version__ as __version__

LIB = Path(__file__).parent


def _check_seed(seed: int | None) -> None:
    """
    Check if the seed is a non-negative integer.

    Parameters
    ----------
    seed : int | None
        The seed value to check.

    Raises
    ------
    ValueError
        If the seed is a negative integer.
    """
    if seed is not None:
        if seed < 0:
            raise ValueError("Seed must be a non-negative integer")


@pl.api.register_expr_namespace("random")
class Random:
    """
    Namespace for random expression functions.

    Parameters
    ----------
    expr : pl.Expr
        The expression to apply the random functions on.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def rand(
        self,
        seed: int | None = None,
    ) -> pl.Expr:
        """
        Generate a random number expression.

        Parameters
        ----------
        seed : int | None, optional
            The seed value for the random number generator, by default None.

        Returns
        -------
        pl.Expr
            The expression with the random number generator applied.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.with_column(pl.col("a").random.rand(seed=42).alias("random"))
        shape: (3, 2)
        ┌─────┬────────────┐
        │ a   │ random     │
        ╞═════╪════════════╡
        │ i64 │ f64        │
        ├─────┼────────────┤
        │ 1   │ 0.37454012 │
        │ 2   │ 0.95071431 │
        │ 3   │ 0.73199394 │
        └─────┴────────────┘
        """
        _check_seed(seed)
        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="rand",
            is_elementwise=True,
            kwargs={"seed": seed},
        )

    def normal(
        self,
        mean: float | None = 0.0,
        std: float | None = 1.0,
        seed: int | None = None,
    ) -> pl.Expr:
        """
        Generate a normal distribution random number expression.

        Parameters
        ----------
        mean : float | None, optional
            The mean of the normal distribution, by default 0.0.
        std : float | None, optional
            The standard deviation of the normal distribution, by default 1.0.
        seed : float | None, optional
            The seed value for the random number generator, by default None.

        Returns
        -------
        pl.Expr
            The expression with the normal distribution random number generator applied.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.with_column(pl.col("a").random.normal(mean=0, std=1, seed=42).alias("normal"))
        shape: (3, 2)
        ┌─────┬────────────┐
        │ a   │ normal     │
        ╞═════╪════════════╡
        │ i64 │ f64        │
        ├─────┼────────────┤
        │ 1   │ 0.49671415 │
        │ 2   │ -0.1382643 │
        │ 3   │ 0.64768854 │
        └─────┴────────────┘
        """
        _check_seed(seed)
        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="normal",
            is_elementwise=True,
            kwargs={"mean": mean, "std": std, "seed": seed},
        )

    def binomial(
        self,
        n: int,
        p: float,
        seed: int | None = None,
    ) -> pl.Expr:
        """
        Generate a binomial distribution random number expression.

        Parameters
        ----------
        n : int
            The number of trials.
        p : float
            The probability of success.
        seed : int | None, optional
            The seed value for the random number generator, by default None.

        Returns
        -------
        pl.Expr
            The expression with the binomial distribution random number generator applied.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> df.with_column(pl.col("a").random.binomial(n=10, p=0.5, seed=42).alias("binomial"))
        shape: (3, 2)
        ┌─────┬────────────┐
        │ a   │ binomial   │
        ╞═════╪════════════╡
        │ i64 │ i64        │
        ├─────┼────────────┤
        │ 1   │ 5          │
        │ 2   │ 5          │
        │ 3   │ 7          │
        └─────┴────────────┘
        """
        _check_seed(seed)
        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="binomial",
            is_elementwise=True,
            kwargs={"n": n, "p": p, "seed": seed},
        )
