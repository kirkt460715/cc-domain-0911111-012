"""Gauge repeatability and reproducibility study lifecycle.

A study registers the crossed design (parts, operators, gauge, replicate count,
allowed collection window) before any observation is bound.  Each accepted
measurement is then attached to one experimental cell (part, operator,
replicate index), reusing the existing calibration correction, policy
evaluation and unit conversion through the normal measurement pipeline.

The design is immutable once collection starts and observations can never be
unbound or silently replaced, so an outlier cannot be deleted and retried as a
fresh experiment.  Missing replicates, extra replicates, record reuse and
deviations from the plan all surface in the completeness result.
"""

from collections import Counter, defaultdict
import datetime as dt
from ..errors import Conflict, InputError, Missing, StateError
from ..values import (
    canonical,
    strict_json,
    digest,
    identifier,
    instant,
    time_text,
    integer,
    require_object,
)
from ..ledger.audit import append
from .grr import Reading, analyze_balanced, cell_diagnostics


class _BindingDeviation(Exception):
    """Carries a plan-deviation reason plus the public error to raise."""

    def __init__(self, reason, error, part=None, operator=None, replicate=None):
        super().__init__(reason)
        self.reason = reason
        self.error = error
        self.part = part
        self.operator = operator
        self.replicate = replicate


class GaugeStudies:
    def __init__(self, database, catalog, pipeline):
        self.db = database
        self.catalog = catalog
        self.pipeline = pipeline
        with self.db.transaction() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS gauge_studies(id TEXT PRIMARY KEY, specification TEXT NOT NULL,
                fingerprint TEXT NOT NULL, state TEXT NOT NULL, revision INTEGER NOT NULL,
                report TEXT, decided TEXT)"""
            )
            con.execute(
                """CREATE TABLE IF NOT EXISTS gauge_bindings(study TEXT NOT NULL, observation TEXT NOT NULL,
                part TEXT NOT NULL, operator TEXT NOT NULL, replicate INTEGER NOT NULL, revision INTEGER NOT NULL,
                PRIMARY KEY(study, observation), UNIQUE(study, part, operator, replicate))"""
            )
            con.execute(
                """CREATE TABLE IF NOT EXISTS gauge_attempts(study TEXT NOT NULL, observation TEXT NOT NULL,
                part TEXT, operator TEXT, replicate INTEGER, reason TEXT NOT NULL, at TEXT NOT NULL,
                PRIMARY KEY(study, observation))"""
            )

    def plan(self, data):
        require_object(
            data,
            {
                "id",
                "instrument",
                "channel",
                "start",
                "end",
                "parts",
                "operators",
                "replicates",
                "tolerance_width",
                "process_variation",
                "sources",
            },
            {
                "id",
                "instrument",
                "channel",
                "start",
                "end",
                "parts",
                "operators",
                "replicates",
            },
        )
        study_id = identifier(data["id"])
        instrument = identifier(data["instrument"])
        channel = identifier(data["channel"])
        start, end = instant(data["start"]), instant(data["end"])
        if end <= start:
            raise InputError("gauge study window must increase")
        parts = self._bounded_identifiers(data["parts"], "parts", 2, 256)
        operators = self._bounded_identifiers(
            data["operators"], "operators", 2, 256
        )
        replicates = integer(data["replicates"], "replicates", 2, 1000)
        self.catalog.channel(instrument, channel)
        self.catalog.bound_policy(instrument, channel)
        gaps = self.catalog.certificates().gaps(instrument, channel, start, end)
        if gaps:
            raise StateError(
                "gauge study requires full calibration coverage",
                details={"channel": channel, "gaps": gaps},
            )
        sources = data.get("sources", [])
        if not isinstance(sources, list) or not sources or len(sources) > 128:
            raise InputError("gauge study sources must be a nonempty bounded list")
        if len(set(sources)) != len(sources):
            raise InputError("duplicate gauge study source")
        sources = sorted(identifier(source) for source in sources)
        for source in sources:
            self.catalog.source(source)
        spec = {
            "id": study_id,
            "instrument": instrument,
            "channel": channel,
            "start": time_text(start),
            "end": time_text(end),
            "parts": parts,
            "operators": operators,
            "replicates": replicates,
            "sources": sources,
            "tolerance_width": self._optional_positive(data, "tolerance_width"),
            "process_variation": self._optional_positive(data, "process_variation"),
        }
        fingerprint = digest(spec)
        with self.db.transaction() as con:
            row = con.execute(
                "SELECT fingerprint,state FROM gauge_studies WHERE id=?",
                (study_id,),
            ).fetchone()
            if row:
                if row["fingerprint"] != fingerprint:
                    raise Conflict("gauge study plan is immutable")
                return self.get(study_id)
            con.execute(
                "INSERT INTO gauge_studies VALUES(?,?,?,'planned',1,NULL,NULL)",
                (study_id, canonical(spec), fingerprint),
            )
            append(con, "gauge_study_planned", study_id, spec)
        return self.get(study_id)

    @staticmethod
    def _bounded_identifiers(value, field, minimum, maximum):
        if not isinstance(value, list) or not minimum <= len(value) <= maximum:
            raise InputError(field + " must contain between two and bounded entries")
        normalized = [identifier(item, field[:-1]) for item in value]
        if len(set(normalized)) != len(normalized):
            raise InputError("duplicate " + field + " entry")
        return sorted(normalized)

    @staticmethod
    def _optional_positive(data, field):
        from ..values import number

        value = data.get(field)
        if value is None:
            return None
        return decimal_or_none(number(value, field, positive=True))

    def get(self, study_id):
        identifier(study_id)
        row = self.db.one("SELECT * FROM gauge_studies WHERE id=?", (study_id,))
        if row is None:
            raise Missing("gauge study not found")
        return {
            "id": study_id,
            "state": row["state"],
            "revision": row["revision"],
            "specification": strict_json(row["specification"]),
            "report": strict_json(row["report"]) if row["report"] else None,
            "decision": strict_json(row["decided"]) if row["decided"] else None,
        }

    def start(self, study_id, expected_revision):
        with self.db.transaction() as con:
            study = self.get(study_id)
            if study["revision"] != expected_revision:
                raise Conflict("gauge study revision changed")
            if study["state"] != "planned":
                raise StateError("only a planned gauge study can start")
            con.execute(
                "UPDATE gauge_studies SET state='collecting',revision=revision+1 WHERE id=?",
                (study_id,),
            )
            append(
                con,
                "gauge_study_started",
                study_id,
                {"revision": expected_revision + 1},
            )
        return self.get(study_id)

    def bind(self, study_id, observation_id):
        identifier(observation_id, "observation")
        # Pre-checks that do not themselves constitute a recorded deviation:
        # an observation already used, or a study in the wrong state.
        study = self.get(study_id)
        if study["state"] != "collecting":
            raise StateError("observations bind only while collecting")
        existing = self.db.one(
            "SELECT study FROM gauge_bindings WHERE observation=?",
            (observation_id,),
        )
        if existing:
            raise Conflict(
                "observation already bound to a gauge study",
                details={"study": existing["study"]},
            )
        attempted = self.db.one(
            "SELECT reason FROM gauge_attempts WHERE study=? AND observation=?",
            (study_id, observation_id),
        )
        if attempted:
            raise Conflict(
                "observation reuse was already recorded as a plan deviation",
                details={"reason": attempted["reason"]},
            )
        try:
            with self.db.transaction() as con:
                bound = self._validate_binding(con, study_id, observation_id)
                con.execute(
                    "INSERT INTO gauge_bindings VALUES(?,?,?,?,?,?)",
                    (
                        study_id,
                        observation_id,
                        bound["part"],
                        bound["operator"],
                        bound["replicate"],
                        bound["revision"],
                    ),
                )
                append(
                    con,
                    "gauge_observation_bound",
                    study_id,
                    {
                        "observation": observation_id,
                        "revision": bound["revision"],
                        "part": bound["part"],
                        "operator": bound["operator"],
                        "replicate": bound["replicate"],
                    },
                )
        except _BindingDeviation as deviation:
            # Persist the rejected attempt in its own committed transaction so
            # the rollback of the failed binding cannot erase the evidence.
            self._persist_attempt(
                study_id,
                observation_id,
                deviation.part,
                deviation.operator,
                deviation.replicate,
                deviation.reason,
            )
            raise deviation.error
        return self.get(study_id)

    def _validate_binding(self, con, study_id, observation_id):
        study = self.get(study_id)
        spec = study["specification"]
        record = self.pipeline.get(observation_id)
        observation = record["raw"]
        measurement = record["measurement"]
        if measurement is None:
            raise _BindingDeviation(
                "no_measurement",
                StateError(
                    "observation has no usable measurement",
                    details={"error": record["error"]},
                ),
            )
        if not measurement["accepted"]:
            raise _BindingDeviation(
                "quality_failure",
                StateError("only accepted measurements can enter a study"),
            )
        if observation["instrument"] != spec["instrument"]:
            raise _BindingDeviation(
                "wrong_instrument",
                Conflict("observation instrument differs from study gauge"),
            )
        if observation["channel"] != spec["channel"]:
            raise _BindingDeviation(
                "wrong_channel",
                Conflict("observation channel differs from study channel"),
            )
        if observation["source"] not in spec["sources"]:
            raise _BindingDeviation(
                "source_outside_plan",
                Conflict("observation source is outside the study sources"),
            )
        event_time = instant(observation["event_time"])
        if not (instant(spec["start"]) <= event_time < instant(spec["end"])):
            raise _BindingDeviation(
                "outside_window",
                StateError("observation event time is outside the study window"),
            )
        part = observation["tags"].get("part")
        operator = observation["tags"].get("operator")
        replicate_text = observation["tags"].get("replicate")
        if part is None or operator is None or replicate_text is None:
            raise _BindingDeviation(
                "missing_design_tags",
                InputError(
                    "observation tags must identify part, operator and replicate"
                ),
            )
        identifier(part, "part tag")
        identifier(operator, "operator tag")
        if part not in spec["parts"]:
            raise _BindingDeviation(
                "unplanned_part",
                Conflict("part is not registered in the study plan"),
                part=part,
                operator=operator,
            )
        if operator not in spec["operators"]:
            raise _BindingDeviation(
                "unplanned_operator",
                Conflict("operator is not registered in the study plan"),
                part=part,
                operator=operator,
            )
        try:
            replicate = int(replicate_text)
        except (TypeError, ValueError):
            raise _BindingDeviation(
                "invalid_replicate",
                InputError("replicate tag must be an integer"),
                part=part,
                operator=operator,
            )
        if not 1 <= replicate <= spec["replicates"]:
            raise _BindingDeviation(
                "replicate_outside_plan",
                Conflict("replicate index is outside the planned count"),
                part=part,
                operator=operator,
                replicate=replicate,
            )
        duplicate_cell = con.execute(
            "SELECT observation FROM gauge_bindings WHERE study=? AND part=? AND operator=? AND replicate=?",
            (study_id, part, operator, replicate),
        ).fetchone()
        if duplicate_cell:
            raise _BindingDeviation(
                "duplicate_cell",
                Conflict(
                    "experimental cell already occupied",
                    details={"observation": duplicate_cell["observation"]},
                ),
                part=part,
                operator=operator,
                replicate=replicate,
            )
        return {
            "part": part,
            "operator": operator,
            "replicate": replicate,
            "revision": record["revision"],
        }

    def _persist_attempt(
        self, study_id, observation_id, part, operator, replicate, reason
    ):
        at = time_text(dt.datetime.now(dt.timezone.utc))
        with self.db.transaction() as con:
            con.execute(
                "INSERT OR IGNORE INTO gauge_attempts VALUES(?,?,?,?,?,?,?)",
                (study_id, observation_id, part, operator, replicate, reason, at),
            )
            append(
                con,
                "gauge_binding_rejected",
                study_id,
                {
                    "observation": observation_id,
                    "reason": reason,
                    "part": part,
                    "operator": operator,
                    "replicate": replicate,
                },
            )

    def attempts(self, study_id):
        self.get(study_id)
        rows = self.db.rows(
            "SELECT * FROM gauge_attempts WHERE study=? ORDER BY at,observation",
            (study_id,),
        )
        return [
            {
                "observation": row["observation"],
                "part": row["part"],
                "operator": row["operator"],
                "replicate": row["replicate"],
                "reason": row["reason"],
                "at": row["at"],
            }
            for row in rows
        ]

    def bindings(self, study_id):
        self.get(study_id)
        rows = self.db.rows(
            """SELECT b.*,o.event_time FROM gauge_bindings b
            JOIN observations o ON o.id=b.observation WHERE b.study=?
            ORDER BY b.part,b.operator,b.replicate,b.observation""",
            (study_id,),
        )
        return [
            {
                "observation": row["observation"],
                "part": row["part"],
                "operator": row["operator"],
                "replicate": row["replicate"],
                "revision": row["revision"],
                "event_time": row["event_time"],
            }
            for row in rows
        ]

    def _load_readings(self, study_id):
        study = self.get(study_id)
        spec = study["specification"]
        rows = self.db.rows(
            """SELECT b.*,r.payload,r.revision AS active_revision FROM gauge_bindings b
            JOIN results r ON r.observation=b.observation AND r.active=1 WHERE b.study=?
            ORDER BY b.part,b.operator,b.replicate""",
            (study_id,),
        )
        readings = []
        stale = []
        for row in rows:
            if row["active_revision"] != row["revision"]:
                stale.append(
                    {
                        "observation": row["observation"],
                        "bound_revision": row["revision"],
                        "current_revision": row["active_revision"],
                    }
                )
            payload = strict_json(row["payload"])
            readings.append(
                Reading(
                    observation=row["observation"],
                    part=row["part"],
                    operator=row["operator"],
                    replicate=row["replicate"],
                    value=self.pipeline_corrected_value(payload),
                    unit=payload["unit"],
                    revision=row["active_revision"],
                    certificate=payload["certificate"],
                    event_time=payload["observation"]["event_time"],
                )
            )
        return spec, readings, stale

    @staticmethod
    def pipeline_corrected_value(payload):
        from ..values import number

        return number(payload["corrected_value"])

    def completeness(self, study_id):
        spec, readings, stale = self._load_readings(study_id)
        deviations = self.attempts(study_id)
        deviation_counts = dict(
            sorted(Counter(item["reason"] for item in deviations).items())
        )
        planned_cells = {
            (part, operator, replicate)
            for part in spec["parts"]
            for operator in spec["operators"]
            for replicate in range(1, spec["replicates"] + 1)
        }
        occupied = Counter(
            (item.part, item.operator, item.replicate) for item in readings
        )
        missing = [
            {"part": part, "operator": operator, "replicate": replicate}
            for part, operator, replicate in sorted(planned_cells - occupied.keys())
        ]
        extras = [
            {
                "part": part,
                "operator": operator,
                "replicate": replicate,
                "count": count,
            }
            for (part, operator, replicate), count in sorted(occupied.items())
            if (part, operator, replicate) not in planned_cells
        ]
        part_counts = Counter(item.part for item in readings)
        operator_counts = Counter(item.operator for item in readings)
        cell_sizes = Counter(
            (item.part, item.operator) for item in readings
        )
        expected_cell_size = spec["replicates"]
        short_cells = [
            {"part": part, "operator": operator, "observed": count}
            for (part, operator), count in sorted(cell_sizes.items())
            if count < expected_cell_size
        ]
        balanced = (
            not missing
            and not extras
            and not stale
            and not deviations
            and len(readings)
            == len(spec["parts"]) * len(spec["operators"]) * spec["replicates"]
            and all(count == expected_cell_size for count in cell_sizes.values())
        )
        certificates = sorted({item.certificate for item in readings})
        units = sorted({item.unit for item in readings})
        issues = []
        if missing:
            issues.append("missing_replicates")
        if extras:
            issues.append("unplanned_replicates")
        if stale:
            issues.append("bound_record_revised")
        if deviations:
            issues.append("plan_deviations")
        if len(units) > 1:
            issues.append("mixed_units")
        if len(certificates) > 1:
            issues.append("multiple_certificates")
        return {
            "id": study_id,
            "planned_cells": len(planned_cells),
            "bound_observations": len(readings),
            "balanced": balanced,
            "missing": missing,
            "extra_cells": extras,
            "short_cells": short_cells,
            "deviations": deviations,
            "deviation_counts": deviation_counts,
            "part_counts": dict(sorted(part_counts.items())),
            "operator_counts": dict(sorted(operator_counts.items())),
            "revised_bindings": stale,
            "units": units,
            "certificates": certificates,
            "issues": issues,
            "lineage": sorted(
                (
                    {
                        "id": item.observation,
                        "revision": item.revision,
                        "part": item.part,
                        "operator": item.operator,
                        "replicate": item.replicate,
                    }
                    for item in readings
                ),
                key=lambda item: item["id"],
            ),
        }

    def analyze(self, study_id):
        spec, readings, stale = self._load_readings(study_id)
        completeness = self.completeness(study_id)
        if not readings:
            raise StateError("gauge study has no bound observations")
        grouped = defaultdict(list)
        for item in readings:
            grouped[(item.part, item.operator)].append(item)
        diagnostics = {
            "overall": cell_diagnostics(readings),
            "by_operator": {
                operator: cell_diagnostics(
                    [item for item in readings if item.operator == operator]
                )
                for operator in spec["operators"]
                if any(item.operator == operator for item in readings)
            },
            "by_part": {
                part: cell_diagnostics(
                    [item for item in readings if item.part == part]
                )
                for part in spec["parts"]
                if any(item.part == part for item in readings)
            },
            "by_cell": {
                f"{part}|{operator}": cell_diagnostics(cell)
                for (part, operator), cell in sorted(grouped.items())
            },
        }
        result = {
            "id": study_id,
            "completeness": completeness,
            "diagnostics": diagnostics,
            "formula_evaluation": None,
        }
        if completeness["balanced"] and not stale:
            result["formula_evaluation"] = analyze_balanced(
                readings,
                tolerance_width=spec["tolerance_width"],
                process_variation=spec["process_variation"],
            )
        else:
            result["diagnostics"]["note"] = (
                "unbalanced or revised design; raw diagnostics only, no formula-based decision"
            )
        result["fingerprint"] = digest(
            {
                "id": study_id,
                "completeness": completeness,
                "formula_evaluation": result["formula_evaluation"],
            }
        )
        return result

    def submit(self, study_id, expected_revision):
        with self.db.transaction() as con:
            study = self.get(study_id)
            if study["revision"] != expected_revision:
                raise Conflict("gauge study revision changed")
            if study["state"] != "collecting":
                raise StateError("only a collecting gauge study can submit")
            report = self.analyze(study_id)
            if report["formula_evaluation"] is None:
                raise StateError(
                    "gauge study is not balanced; freeze diagnostics instead of a formula result",
                    details=report["completeness"],
                )
            con.execute(
                "UPDATE gauge_studies SET state='review',revision=revision+1,report=? WHERE id=?",
                (canonical(report), study_id),
            )
            append(
                con,
                "gauge_study_submitted",
                study_id,
                {"revision": expected_revision + 1, "report": report["fingerprint"]},
            )
        return self.get(study_id)

    def freeze_diagnostics(self, study_id, expected_revision):
        """Freeze an unbalanced study without a formula-based judgement."""
        with self.db.transaction() as con:
            study = self.get(study_id)
            if study["revision"] != expected_revision:
                raise Conflict("gauge study revision changed")
            if study["state"] != "collecting":
                raise StateError("only a collecting gauge study can freeze")
            report = self.analyze(study_id)
            if report["formula_evaluation"] is not None:
                raise StateError(
                    "balanced studies submit a formula evaluation, not raw diagnostics"
                )
            con.execute(
                "UPDATE gauge_studies SET state='frozen_diagnostics',revision=revision+1,report=? WHERE id=?",
                (canonical(report), study_id),
            )
            append(
                con,
                "gauge_study_frozen",
                study_id,
                {"revision": expected_revision + 1, "report": report["fingerprint"]},
            )
        return self.get(study_id)

    def decide(self, study_id, expected_revision, reviewer, approved, reason):
        from ..values import label

        identifier(reviewer, "reviewer")
        reason = label(reason, "reason")
        if not isinstance(approved, bool):
            raise InputError("decision must be boolean")
        with self.db.transaction() as con:
            study = self.get(study_id)
            if study["revision"] != expected_revision:
                raise Conflict("gauge study revision changed")
            if study["state"] != "review":
                raise StateError("gauge study is not awaiting review")
            if approved:
                current = self.analyze(study_id)
                if (
                    current["formula_evaluation"] is None
                    or current["fingerprint"] != study["report"]["fingerprint"]
                ):
                    raise Conflict(
                        "gauge study data changed during review; reject and investigate"
                    )
            decision = {
                "reviewer": reviewer,
                "approved": approved,
                "reason": reason,
                "report": study["report"]["fingerprint"],
            }
            con.execute(
                "UPDATE gauge_studies SET state=?,revision=revision+1,decided=? WHERE id=?",
                (
                    "approved" if approved else "collecting",
                    canonical(decision),
                    study_id,
                ),
            )
            append(con, "gauge_study_decided", study_id, decision)
        return self.get(study_id)


def decimal_or_none(value):
    from ..values import decimal_text

    return decimal_text(value) if value is not None else None
