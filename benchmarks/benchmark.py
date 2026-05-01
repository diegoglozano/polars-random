"""
Benchmark `polars-random` against `numpy.random`.

Run with::

    uv run --with numpy --no-sync python benchmarks/benchmark.py

Optional flags::

    --sizes 10000,100000,1000000,10000000   # comma-separated row counts
    --repeats 7                              # timing repeats per case (best-of)
    --inner 3                                # inner timeit loops per repeat
    --output benchmarks/results.md           # markdown table written here
    --json benchmarks/results.json           # raw timings for downstream tooling

The script is deterministic: every distribution is drawn with ``seed=42`` and the
same ``size`` argument so successive runs produce numerically identical columns
and the only thing varying is wall-clock time.

Scenarios per distribution / size:

* ``numpy``                 -- ``np.random.default_rng(42).<dist>(..., size=N)``
* ``numpy -> pl.Series``    -- numpy draw + ``pl.Series(...)`` round-trip (the
  cost paid when using numpy from inside a polars pipeline).
* ``polars_random eager``   -- ``pr.<dist>(..., size=N, seed=42)`` top-level API.
* ``polars_random expr``    -- ``df.with_columns(pr.<dist>(..., seed=42))`` on a
  pre-existing N-row frame; this is the typical "add a random column" usage.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import timeit
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

import polars_random as pr


@dataclass
class Case:
    distribution: str
    scenario: str
    size: int
    times: list[float] = field(default_factory=list)

    @property
    def best(self) -> float:
        return min(self.times)

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    def to_dict(self) -> dict:
        return {
            "distribution": self.distribution,
            "scenario": self.scenario,
            "size": self.size,
            "times_seconds": self.times,
            "best_seconds": self.best,
            "median_seconds": self.median,
        }


def time_callable(fn, *, repeats: int, inner: int) -> list[float]:
    """Return per-call seconds for `repeats` measurements of `inner` calls each."""
    # Warm up once to avoid first-call lazy initialisation skew.
    fn()
    raw = timeit.repeat(fn, repeat=repeats, number=inner)
    return [t / inner for t in raw]


def _make_anchor_df(n: int) -> pl.DataFrame:
    """A length-N frame the random expression can attach to (column contents irrelevant)."""
    return pl.DataFrame({"_": pl.zeros(n, eager=True)})


def build_cases(sizes: list[int]) -> list[tuple[str, str, int, Callable[[], object]]]:
    """Return ``(distribution, scenario, size, fn)`` tuples ready to be timed."""
    cases: list[tuple[str, str, int, Callable[[], object]]] = []

    for n in sizes:
        anchor = _make_anchor_df(n)

        # ---- uniform ----------------------------------------------------------------
        def numpy_uniform(n=n):
            return np.random.default_rng(42).uniform(0.0, 1.0, size=n)

        def numpy_uniform_series(n=n):
            arr = np.random.default_rng(42).uniform(0.0, 1.0, size=n)
            return pl.Series("uniform", arr)

        def pr_uniform_eager(n=n):
            return pr.rand(low=0.0, high=1.0, size=n, seed=42)

        def pr_uniform_expr(df=anchor):
            return df.with_columns(pr.rand(low=0.0, high=1.0, seed=42).alias("uniform"))

        cases += [
            ("uniform", "numpy", n, numpy_uniform),
            ("uniform", "numpy -> pl.Series", n, numpy_uniform_series),
            ("uniform", "polars_random eager", n, pr_uniform_eager),
            ("uniform", "polars_random expr", n, pr_uniform_expr),
        ]

        # ---- normal -----------------------------------------------------------------
        def numpy_normal(n=n):
            return np.random.default_rng(42).normal(0.0, 1.0, size=n)

        def numpy_normal_series(n=n):
            arr = np.random.default_rng(42).normal(0.0, 1.0, size=n)
            return pl.Series("normal", arr)

        def pr_normal_eager(n=n):
            return pr.normal(mean=0.0, std=1.0, size=n, seed=42)

        def pr_normal_expr(df=anchor):
            return df.with_columns(pr.normal(mean=0.0, std=1.0, seed=42).alias("normal"))

        cases += [
            ("normal", "numpy", n, numpy_normal),
            ("normal", "numpy -> pl.Series", n, numpy_normal_series),
            ("normal", "polars_random eager", n, pr_normal_eager),
            ("normal", "polars_random expr", n, pr_normal_expr),
        ]

        # ---- binomial ---------------------------------------------------------------
        def numpy_binomial(n=n):
            return np.random.default_rng(42).binomial(100, 0.5, size=n)

        def numpy_binomial_series(n=n):
            arr = np.random.default_rng(42).binomial(100, 0.5, size=n)
            return pl.Series("binomial", arr)

        def pr_binomial_eager(n=n):
            return pr.binomial(n=100, p=0.5, size=n, seed=42)

        def pr_binomial_expr(df=anchor):
            return df.with_columns(pr.binomial(n=100, p=0.5, seed=42).alias("binomial"))

        cases += [
            ("binomial", "numpy", n, numpy_binomial),
            ("binomial", "numpy -> pl.Series", n, numpy_binomial_series),
            ("binomial", "polars_random eager", n, pr_binomial_eager),
            ("binomial", "polars_random expr", n, pr_binomial_expr),
        ]

        # ---- randint ----------------------------------------------------------------
        def numpy_randint(n=n):
            return np.random.default_rng(42).integers(0, 1000, size=n)

        def numpy_randint_series(n=n):
            arr = np.random.default_rng(42).integers(0, 1000, size=n)
            return pl.Series("randint", arr)

        def pr_randint_eager(n=n):
            return pr.randint(low=0, high=1000, size=n, seed=42)

        def pr_randint_expr(df=anchor):
            return df.with_columns(pr.randint(low=0, high=1000, seed=42).alias("randint"))

        cases += [
            ("randint", "numpy", n, numpy_randint),
            ("randint", "numpy -> pl.Series", n, numpy_randint_series),
            ("randint", "polars_random eager", n, pr_randint_eager),
            ("randint", "polars_random expr", n, pr_randint_expr),
        ]

    return cases


def format_seconds(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:7.1f} us"
    if s < 1.0:
        return f"{s * 1e3:7.2f} ms"
    return f"{s:7.3f} s"


def format_throughput(seconds: float, n: int) -> str:
    if seconds == 0:
        return "n/a"
    rate = n / seconds
    if rate > 1e9:
        return f"{rate / 1e9:5.2f} G/s"
    if rate > 1e6:
        return f"{rate / 1e6:5.1f} M/s"
    if rate > 1e3:
        return f"{rate / 1e3:5.1f} K/s"
    return f"{rate:5.1f} /s"


def render_markdown(results: list[Case], info: dict) -> str:
    lines: list[str] = []
    lines.append("# polars-random benchmark results")
    lines.append("")
    lines.append("Generated by `benchmarks/benchmark.py`. All draws use `seed=42`.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    for k, v in info.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "Each row reports the *best* of `--repeats` timed runs (each run averages "
        "`--inner` calls). `polars_random` numbers below are throughput against the "
        "numpy baseline of the same size."
    )
    lines.append("")

    distributions: list[str] = []
    for c in results:
        if c.distribution not in distributions:
            distributions.append(c.distribution)

    for dist in distributions:
        lines.append(f"### `{dist}`")
        lines.append("")
        lines.append("| Size | Scenario | Best | Median | Throughput | Speedup vs numpy |")
        lines.append("|---|---|---:|---:|---:|---:|")

        sizes_for_dist = sorted({c.size for c in results if c.distribution == dist})
        for size in sizes_for_dist:
            cases = [c for c in results if c.distribution == dist and c.size == size]
            numpy_case = next((c for c in cases if c.scenario == "numpy"), None)
            for c in cases:
                speedup = ""
                if numpy_case is not None and c.best > 0:
                    ratio = numpy_case.best / c.best
                    speedup = f"{ratio:.2f}x"
                lines.append(
                    f"| {size:,} | {c.scenario} | {format_seconds(c.best)} | "
                    f"{format_seconds(c.median)} | "
                    f"{format_throughput(c.best, size)} | {speedup} |"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        default="10000,100000,1000000,10000000",
        help="Comma-separated list of row counts to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--inner", type=int, default=3)
    parser.add_argument(
        "--output",
        default="benchmarks/results.md",
        help="Markdown file to write the results table to.",
    )
    parser.add_argument(
        "--json",
        default="benchmarks/results.json",
        help="JSON file to dump raw per-run timings to.",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    cases = build_cases(sizes)

    print(
        f"Running {len(cases)} cases (repeats={args.repeats}, inner={args.inner})...",
        flush=True,
    )
    results: list[Case] = []
    for distribution, scenario, size, fn in cases:
        case = Case(distribution=distribution, scenario=scenario, size=size)
        case.times = time_callable(fn, repeats=args.repeats, inner=args.inner)
        results.append(case)
        print(
            f"  {distribution:>9} | {scenario:<22} | size={size:>10,} | "
            f"best={format_seconds(case.best)} | "
            f"throughput={format_throughput(case.best, size)}",
            flush=True,
        )

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "polars": pl.__version__,
        "polars_random": pr.__version__,
        "numpy": np.__version__,
        "repeats": args.repeats,
        "inner": args.inner,
        "sizes": sizes,
    }

    out_md = Path(args.output)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(results, info))
    print(f"\nWrote markdown to {out_md}")

    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {"info": info, "results": [c.to_dict() for c in results]},
            indent=2,
        )
    )
    print(f"Wrote raw timings to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
