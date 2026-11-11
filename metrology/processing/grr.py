"""Crossed gauge repeatability and reproducibility statistics.

The formulas follow the classical two-way crossed design with replication
(parts x operators, each cell measured r times).  Every numeric step uses
bounded ``Decimal`` arithmetic so that frozen reports are reproducible across
runtimes and free of binary floating point drift.

Two distinct scopes are kept separate on purpose:

* tolerance ratios (``*_vs_tolerance``) divide six-sigma study variation by the
  engineering tolerance width;
* process ratios (``*_vs_process``) divide by an externally supplied process
  variation.  Part-to-part variation is never silently treated as a process
  standard deviation.

When the data are not perfectly balanced, no formula-based pass/fail decision
or variance component is produced; callers receive raw diagnostics only.
"""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from ..errors import InputError
from ..values import number, decimal_text, integer

PRECISION = 40
# Standard normal quantile used to convert a standard deviation into the
# six-sigma study variation convention of measurement system analysis.
SIX_SIGMA = Decimal("6")
D0 = Decimal(0)
D1 = Decimal(1)


# Selected upper 5% critical values of the F distribution (numerator first).
# Only degrees of freedom reachable by small, bounded gauge studies are
# tabulated; a missing combination means the pooling decision is inconclusive.
F_TABLE_05 = {
    (1, 1): "161.45", (1, 2): "18.51", (1, 3): "10.13", (1, 4): "7.71",
    (1, 5): "6.61", (1, 6): "5.99", (1, 7): "5.59", (1, 8): "5.32",
    (1, 9): "5.12", (1, 10): "4.96", (1, 12): "4.75", (1, 15): "4.54",
    (1, 20): "4.35", (1, 24): "4.26", (1, 30): "4.17", (1, 40): "4.08",
    (1, 60): "4.00", (1, 120): "3.92", (1, 1000000): "3.84",
    (2, 1): "199.50", (2, 2): "19.00", (2, 3): "9.55", (2, 4): "6.94",
    (2, 5): "5.79", (2, 6): "5.14", (2, 7): "4.74", (2, 8): "4.46",
    (2, 9): "4.26", (2, 10): "4.10", (2, 12): "3.89", (2, 15): "3.68",
    (2, 20): "3.49", (2, 24): "3.40", (2, 30): "3.32", (2, 40): "3.23",
    (2, 60): "3.15", (2, 120): "3.07", (2, 1000000): "3.00",
    (3, 1): "215.71", (3, 2): "19.16", (3, 3): "9.28", (3, 4): "6.59",
    (3, 5): "5.41", (3, 6): "4.76", (3, 7): "4.35", (3, 8): "4.07",
    (3, 9): "3.86", (3, 10): "3.71", (3, 12): "3.49", (3, 15): "3.29",
    (3, 20): "3.10", (3, 24): "3.01", (3, 30): "2.92", (3, 40): "2.84",
    (3, 60): "2.76", (3, 120): "2.68", (3, 1000000): "2.60",
    (4, 1): "224.58", (4, 2): "19.25", (4, 3): "9.12", (4, 4): "6.39",
    (4, 5): "5.19", (4, 6): "4.53", (4, 7): "4.12", (4, 8): "3.84",
    (4, 9): "3.63", (4, 10): "3.48", (4, 12): "3.26", (4, 15): "3.06",
    (4, 20): "2.87", (4, 24): "2.78", (4, 30): "2.69", (4, 40): "2.61",
    (4, 60): "2.53", (4, 120): "2.45", (4, 1000000): "2.37",
    (5, 1): "230.16", (5, 2): "19.30", (5, 3): "9.01", (5, 4): "6.26",
    (5, 5): "5.05", (5, 6): "4.39", (5, 7): "3.97", (5, 8): "3.69",
    (5, 9): "3.48", (5, 10): "3.33", (5, 12): "3.11", (5, 15): "2.90",
    (5, 20): "2.71", (5, 24): "2.62", (5, 30): "2.53", (5, 40): "2.45",
    (5, 60): "2.37", (5, 120): "2.29", (5, 1000000): "2.21",
    (6, 1): "233.99", (6, 2): "19.33", (6, 3): "8.94", (6, 4): "6.16",
    (6, 5): "4.95", (6, 6): "4.28", (6, 7): "3.87", (6, 8): "3.58",
    (6, 9): "3.37", (6, 10): "3.22", (6, 12): "3.00", (6, 15): "2.79",
    (6, 20): "2.60", (6, 24): "2.51", (6, 30): "2.42", (6, 40): "2.34",
    (6, 60): "2.25", (6, 120): "2.18", (6, 1000000): "2.10",
    (7, 1): "236.77", (7, 2): "19.35", (7, 3): "8.89", (7, 4): "6.09",
    (7, 5): "4.88", (7, 6): "4.21", (7, 7): "3.79", (7, 8): "3.50",
    (7, 9): "3.29", (7, 10): "3.14", (7, 12): "2.91", (7, 15): "2.71",
    (7, 20): "2.51", (7, 24): "2.42", (7, 30): "2.33", (7, 40): "2.25",
    (7, 60): "2.17", (7, 120): "2.09", (7, 1000000): "2.01",
    (8, 1): "238.88", (8, 2): "19.37", (8, 3): "8.85", (8, 4): "6.04",
    (8, 5): "4.82", (8, 6): "4.15", (8, 7): "3.73", (8, 8): "3.44",
    (8, 9): "3.23", (8, 10): "3.07", (8, 12): "2.85", (8, 15): "2.64",
    (8, 20): "2.45", (8, 24): "2.36", (8, 30): "2.27", (8, 40): "2.18",
    (8, 60): "2.10", (8, 120): "2.02", (8, 1000000): "1.94",
    (9, 1): "240.54", (9, 2): "19.38", (9, 3): "8.81", (9, 4): "6.00",
    (9, 5): "4.77", (9, 6): "4.10", (9, 7): "3.68", (9, 8): "3.39",
    (9, 9): "3.18", (9, 10): "3.02", (9, 12): "2.80", (9, 15): "2.59",
    (9, 20): "2.39", (9, 24): "2.30", (9, 30): "2.21", (9, 40): "2.12",
    (9, 60): "2.04", (9, 120): "1.96", (9, 1000000): "1.88",
    (10, 1): "241.88", (10, 2): "19.40", (10, 3): "8.79", (10, 4): "5.96",
    (10, 5): "4.74", (10, 6): "4.06", (10, 7): "3.64", (10, 8): "3.35",
    (10, 9): "3.14", (10, 10): "2.98", (10, 12): "2.75", (10, 15): "2.54",
    (10, 20): "2.35", (10, 24): "2.25", (10, 30): "2.16", (10, 40): "2.08",
    (10, 60): "1.99", (10, 120): "1.91", (10, 1000000): "1.83",
    (12, 1): "243.91", (12, 2): "19.41", (12, 3): "8.74", (12, 4): "5.91",
    (12, 5): "4.68", (12, 6): "4.00", (12, 7): "3.57", (12, 8): "3.28",
    (12, 9): "3.07", (12, 10): "2.91", (12, 12): "2.69", (12, 15): "2.48",
    (12, 20): "2.28", (12, 24): "2.18", (12, 30): "2.09", (12, 40): "2.00",
    (12, 60): "1.92", (12, 120): "1.83", (12, 1000000): "1.75",
    (15, 1): "245.95", (15, 2): "19.43", (15, 3): "8.70", (15, 4): "5.86",
    (15, 5): "4.62", (15, 6): "3.94", (15, 7): "3.51", (15, 8): "3.22",
    (15, 9): "3.01", (15, 10): "2.85", (15, 12): "2.62", (15, 15): "2.40",
    (15, 20): "2.20", (15, 24): "2.11", (15, 30): "2.01", (15, 40): "1.92",
    (15, 60): "1.84", (15, 120): "1.75", (15, 1000000): "1.67",
    (20, 1): "248.01", (20, 2): "19.45", (20, 3): "8.66", (20, 4): "5.80",
    (20, 5): "4.56", (20, 6): "3.87", (20, 7): "3.44", (20, 8): "3.15",
    (20, 9): "2.94", (20, 10): "2.77", (20, 12): "2.54", (20, 15): "2.33",
    (20, 20): "2.12", (20, 24): "2.03", (20, 30): "1.93", (20, 40): "1.84",
    (20, 60): "1.75", (20, 120): "1.66", (20, 1000000): "1.57",
    (24, 1): "249.05", (24, 2): "19.45", (24, 3): "8.64", (24, 4): "5.77",
    (24, 5): "4.53", (24, 6): "3.84", (24, 7): "3.41", (24, 8): "3.12",
    (24, 9): "2.90", (24, 10): "2.74", (24, 12): "2.51", (24, 15): "2.29",
    (24, 20): "2.08", (24, 24): "1.98", (24, 30): "1.89", (24, 40): "1.79",
    (24, 60): "1.70", (24, 120): "1.61", (24, 1000000): "1.52",
    (30, 1): "250.10", (30, 2): "19.46", (30, 3): "8.62", (30, 4): "5.75",
    (30, 5): "4.50", (30, 6): "3.81", (30, 7): "3.38", (30, 8): "3.08",
    (30, 9): "2.86", (30, 10): "2.70", (30, 12): "2.47", (30, 15): "2.25",
    (30, 20): "2.04", (30, 24): "1.94", (30, 30): "1.84", (30, 40): "1.74",
    (30, 60): "1.65", (30, 120): "1.55", (30, 1000000): "1.46",
    (40, 1): "251.14", (40, 2): "19.47", (40, 3): "8.59", (40, 4): "5.72",
    (40, 5): "4.46", (40, 6): "3.77", (40, 7): "3.34", (40, 8): "3.04",
    (40, 9): "2.83", (40, 10): "2.66", (40, 12): "2.43", (40, 15): "2.20",
    (40, 20): "1.99", (40, 24): "1.89", (40, 30): "1.79", (40, 40): "1.69",
    (40, 60): "1.59", (40, 120): "1.50", (40, 1000000): "1.39",
    (60, 1): "252.20", (60, 2): "19.48", (60, 3): "8.57", (60, 4): "5.69",
    (60, 5): "4.43", (60, 6): "3.74", (60, 7): "3.30", (60, 8): "3.01",
    (60, 9): "2.79", (60, 10): "2.62", (60, 12): "2.39", (60, 15): "2.16",
    (60, 20): "1.95", (60, 24): "1.84", (60, 30): "1.74", (60, 40): "1.64",
    (60, 60): "1.53", (60, 120): "1.43", (60, 1000000): "1.32",
    (120, 1): "253.25", (120, 2): "19.49", (120, 3): "8.55", (120, 4): "5.66",
    (120, 5): "4.40", (120, 6): "3.70", (120, 7): "3.27", (120, 8): "2.97",
    (120, 9): "2.75", (120, 10): "2.58", (120, 12): "2.34", (120, 15): "2.11",
    (120, 20): "1.90", (120, 24): "1.79", (120, 30): "1.68", (120, 40): "1.58",
    (120, 60): "1.47", (120, 120): "1.35", (120, 1000000): "1.22",
}


@dataclass(frozen=True)
class Reading:
    """One accepted measurement positioned inside the crossed design."""

    observation: str
    part: str
    operator: str
    replicate: int
    value: Decimal
    unit: str
    revision: int
    certificate: str
    event_time: str


def f_critical(numerator_df, denominator_df):
    """Return the tabulated 5% F critical value, or ``None`` if unavailable."""
    text = F_TABLE_05.get((numerator_df, denominator_df))
    return Decimal(text) if text is not None else None


def _q(value):
    return decimal_text(value.quantize(Decimal("0.000001")))


def _safe_f(numerator, denominator):
    """F ratio that survives degenerate zero-spread data.

    A zero numerator is a genuine zero statistic; a zero denominator with a
    positive numerator is undefined (infinite in the limit) and is reported as
    ``None`` rather than raising.
    """
    if numerator == 0:
        return D0
    if denominator == 0:
        return None
    return numerator / denominator


def _interaction_significant(ms_interaction, ms_error, critical):
    if ms_interaction == 0:
        return False
    if ms_error == 0:
        return True
    statistic = ms_interaction / ms_error
    if critical is None:
        return None
    return statistic > critical


def _variance_component(estimate):
    """Apply the documented non-negative truncation to an ANOVA estimate."""
    truncated = estimate < 0
    return {
        "estimate": decimal_text(estimate),
        "variance": decimal_text(max(estimate, D0)),
        "truncated_to_zero": truncated,
    }


def cell_diagnostics(cells):
    """Raw, formula-free diagnostics usable for balanced and unbalanced data."""
    values = [item.value for item in cells]
    with localcontext() as ctx:
        ctx.prec = PRECISION
        count = Decimal(len(values))
        mean = sum(values, D0) / count
        spread = sum(((v - mean) ** 2 for v in values), D0) / count
        deviations = sorted(abs(v - mean) for v in values)
        median_abs = (
            deviations[len(deviations) // 2]
            if len(deviations) % 2
            else (deviations[len(deviations) // 2 - 1] + deviations[len(deviations) // 2]) / 2
        )
    return {
        "count": len(values),
        "mean": decimal_text(mean),
        "minimum": decimal_text(min(values)),
        "maximum": decimal_text(max(values)),
        "range": decimal_text(max(values) - min(values)),
        "standard_deviation": decimal_text(spread.sqrt()),
        "median_absolute_deviation": decimal_text(median_abs),
    }


def analyze_balanced(readings, *, tolerance_width=None, process_variation=None):
    """Compute the crossed ANOVA and variance components.

    ``readings`` must describe a fully balanced design: every part/operator
    cell has the same number of replicates and every value shares one unit.
    """
    if not readings:
        raise InputError("gauge study requires at least one reading")
    units = {item.unit for item in readings}
    if len(units) != 1:
        raise InputError("gauge study readings must share one unit")
    unit = next(iter(units))
    parts = sorted({item.part for item in readings})
    operators = sorted({item.operator for item in readings})
    p, o = len(parts), len(operators)
    grid = {
        (item.part, item.operator): []
        for item in readings
    }
    for item in readings:
        grid[(item.part, item.operator)].append(item)
    replicates = {len(cell) for cell in grid.values()}
    if len(grid) != p * o or len(replicates) != 1:
        raise InputError("analyze_balanced requires a complete crossed design")
    r = replicates.pop()
    integer(r, "replicates", 1, 10000)
    if r < 2:
        raise InputError(
            "crossed gauge study requires at least two replicates per cell "
            "to estimate repeatability"
        )

    width = None
    if tolerance_width is not None:
        width = number(tolerance_width, "tolerance_width", positive=True)
    process = None
    if process_variation is not None:
        process = number(process_variation, "process_variation", positive=True)

    with localcontext() as ctx:
        ctx.prec = PRECISION
        n = Decimal(p * o * r)
        grand = sum((item.value for item in readings), D0) / n

        part_means = {
            part: sum(
                (item.value for item in readings if item.part == part), D0
            ) / Decimal(o * r)
            for part in parts
        }
        operator_means = {
            operator: sum(
                (item.value for item in readings if item.operator == operator), D0
            ) / Decimal(p * r)
            for operator in operators
        }
        cell_means = {
            key: sum((item.value for item in cell), D0) / Decimal(r)
            for key, cell in grid.items()
        }

        ss_part = Decimal(o * r) * sum(
            ((mean - grand) ** 2 for mean in part_means.values()), D0
        )
        ss_operator = Decimal(p * r) * sum(
            ((mean - grand) ** 2 for mean in operator_means.values()), D0
        )
        ss_interaction = Decimal(r) * sum(
            (
                (
                    cell_means[(part, operator)]
                    - part_means[part]
                    - operator_means[operator]
                    + grand
                )
                ** 2
                for part in parts
                for operator in operators
            ),
            D0,
        )
        ss_error = sum(
            ((item.value - cell_means[(item.part, item.operator)]) ** 2 for item in readings),
            D0,
        )
        ss_total = sum(((item.value - grand) ** 2 for item in readings), D0)

        df_part, df_operator, df_interaction, df_error = p - 1, o - 1, (p - 1) * (o - 1), p * o * (r - 1)
        ms_part = ss_part / Decimal(df_part)
        ms_operator = ss_operator / Decimal(df_operator)
        ms_interaction = ss_interaction / Decimal(df_interaction)
        ms_error = ss_error / Decimal(df_error)

        f_part = _safe_f(ms_part, ms_interaction)
        f_operator = _safe_f(ms_operator, ms_interaction)
        f_interaction = _safe_f(ms_interaction, ms_error)
        fc_part = f_critical(df_part, df_interaction)
        fc_operator = f_critical(df_operator, df_interaction)
        fc_interaction = f_critical(df_interaction, df_error)

        interaction_significant = _interaction_significant(
            ms_interaction, ms_error, fc_interaction
        )

        # Expected mean square structure of the crossed random-effects model.
        variance_repeat = ms_error
        if interaction_significant is False:
            # Pool interaction into repeatability: SS and df are combined and
            # the pooled mean square becomes the new error term.
            pooled_ms = (ss_interaction + ss_error) / Decimal(
                df_interaction + df_error
            )
            variance_repeat = pooled_ms
            variance_interaction = D0
            pooled = True
            error_label = "repeatability"
            error_df = df_interaction + df_error
            error_ms = pooled_ms
        else:
            variance_interaction = (
                (ms_interaction - ms_error) / Decimal(r)
            )
            pooled = False
            error_label = "repeatability"
            error_df = df_error
            error_ms = ms_error

        variance_part = (
            ms_part - ms_interaction
        ) / Decimal(o * r)
        variance_operator = (
            ms_operator - ms_interaction
        ) / Decimal(p * r)

        components = {
            "part_to_part": _variance_component(variance_part),
            "operator": _variance_component(variance_operator),
            "interaction": _variance_component(variance_interaction),
            "repeatability": _variance_component(variance_repeat),
        }
        var_part = max(variance_part, D0)
        var_operator = max(variance_operator, D0)
        var_interaction = max(variance_interaction, D0)
        var_repeat = max(variance_repeat, D0)
        var_reproducibility = var_operator + var_interaction
        var_grr = var_repeat + var_reproducibility
        var_total = var_grr + var_part

        def sd(variance):
            return variance.sqrt()

        sd_part, sd_operator = sd(var_part), sd(var_operator)
        sd_interaction, sd_repeat = sd(var_interaction), sd(var_repeat)
        sd_repro, sd_grr, sd_total = (
            sd(var_reproducibility),
            sd(var_grr),
            sd(var_total),
        )

        def percent(numerator_sd, denominator):
            if denominator is None or denominator == 0 or sd_total == 0:
                return None
            return _q(SIX_SIGMA * numerator_sd / denominator)

        tolerance_denominator = SIX_SIGMA * width if width is not None else None
        process_denominator = SIX_SIGMA * process if process is not None else None
        contributions = {
            "repeatability": {
                "standard_deviation": decimal_text(sd_repeat),
                "study_variation": decimal_text(SIX_SIGMA * sd_repeat),
                "percent_vs_tolerance": percent(sd_repeat, tolerance_denominator),
                "percent_vs_process": percent(sd_repeat, process_denominator),
            },
            "reproducibility": {
                "standard_deviation": decimal_text(sd_repro),
                "study_variation": decimal_text(SIX_SIGMA * sd_repro),
                "operator_standard_deviation": decimal_text(sd_operator),
                "interaction_standard_deviation": decimal_text(sd_interaction),
                "percent_vs_tolerance": percent(sd_repro, tolerance_denominator),
                "percent_vs_process": percent(sd_repro, process_denominator),
            },
            "gauge": {
                "standard_deviation": decimal_text(sd_grr),
                "study_variation": decimal_text(SIX_SIGMA * sd_grr),
                "percent_vs_tolerance": percent(sd_grr, tolerance_denominator),
                "percent_vs_process": percent(sd_grr, process_denominator),
            },
            "part_to_part": {
                "standard_deviation": decimal_text(sd_part),
                "study_variation": decimal_text(SIX_SIGMA * sd_part),
                "percent_vs_tolerance": percent(sd_part, tolerance_denominator),
                "percent_vs_process": percent(sd_part, process_denominator),
            },
            "total": {
                "standard_deviation": decimal_text(sd_total),
                "study_variation": decimal_text(SIX_SIGMA * sd_total),
                "percent_vs_tolerance": percent(sd_total, tolerance_denominator),
                "percent_vs_process": percent(sd_total, process_denominator),
            },
        }

        anova_table = [
            {
                "source": "part",
                "degrees_of_freedom": df_part,
                "sum_of_squares": decimal_text(ss_part),
                "mean_square": decimal_text(ms_part),
                "f_statistic": decimal_text(f_part) if f_part is not None else None,
                "f_critical_05": decimal_text(fc_part) if fc_part is not None else None,
            },
            {
                "source": "operator",
                "degrees_of_freedom": df_operator,
                "sum_of_squares": decimal_text(ss_operator),
                "mean_square": decimal_text(ms_operator),
                "f_statistic": decimal_text(f_operator)
                if f_operator is not None
                else None,
                "f_critical_05": decimal_text(fc_operator)
                if fc_operator is not None
                else None,
            },
        ]
        if not pooled:
            anova_table.append(
                {
                    "source": "interaction",
                    "degrees_of_freedom": df_interaction,
                    "sum_of_squares": decimal_text(ss_interaction),
                    "mean_square": decimal_text(ms_interaction),
                    "f_statistic": decimal_text(f_interaction)
                    if f_interaction is not None
                    else None,
                    "f_critical_05": decimal_text(fc_interaction)
                    if fc_interaction is not None
                    else None,
                }
            )
        anova_table.append(
            {
                "source": error_label,
                "degrees_of_freedom": error_df,
                "sum_of_squares": decimal_text(ss_error + ss_interaction)
                if pooled
                else decimal_text(ss_error),
                "mean_square": decimal_text(error_ms),
                "f_statistic": None,
                "f_critical_05": None,
            }
        )
        anova_table.append(
            {
                "source": "total",
                "degrees_of_freedom": p * o * r - 1,
                "sum_of_squares": decimal_text(ss_total),
                "mean_square": None,
                "f_statistic": None,
                "f_critical_05": None,
            }
        )

    return {
        "design": {
            "parts": parts,
            "operators": operators,
            "replicates": r,
            "part_count": p,
            "operator_count": o,
            "balanced": True,
        },
        "unit": unit,
        "anova": anova_table,
        "interaction": {
            "f_statistic": decimal_text(f_interaction)
            if f_interaction is not None
            else None,
            "f_critical_05": decimal_text(fc_interaction)
            if fc_interaction is not None
            else None,
            "significant": interaction_significant,
            "pooled_into_repeatability": pooled,
            "basis": (
                "F interaction below tabulated 5% critical value; interaction SS pooled with residual SS"
                if pooled
                else "interaction retained as a separate variance component"
            ),
        },
        "variance_components": components,
        "contributions": contributions,
        "tolerance_width": decimal_text(width) if width is not None else None,
        "process_variation": decimal_text(process) if process is not None else None,
    }
