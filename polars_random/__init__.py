from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

from polars_random._internal import __version__ as __version__

LIB = Path(__file__).parent


@pl.api.register_expr_namespace("random")
class Random:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def rand(
        self,
        seed: int | None = None,
    ) -> pl.Expr:
        if seed is not None:
            if seed < 0:
                raise ValueError("Seed must be a non-negative integer")
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
        seed: float | None = None,
    ) -> pl.Expr:
        return register_plugin_function(
            args=[self._expr],
            plugin_path=LIB,
            function_name="normal",
            is_elementwise=True,
            kwargs={"mean": mean, "std": std, "seed": seed},
        )
