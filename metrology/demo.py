import json
import tempfile
from pathlib import Path
from .service import Service
from .ingress.readers import read_jsonl
from .output.snapshots import verify_export


def demonstration():
    with tempfile.TemporaryDirectory(prefix="liangheng-") as directory:
        root = Path(directory)
        with Service(root / "lab.sqlite3") as app:
            app.configuration.apply(
                {
                    "id": "demo-config",
                    "instruments": [
                        {
                            "id": "press-1",
                            "label": "Synthetic pressure instrument",
                            "channels": {"p": "kPa"},
                        }
                    ],
                    "sources": [{"id": "bench", "allowed_lateness": 0}],
                    "certificates": [
                        {
                            "id": "cert-1",
                            "instrument": "press-1",
                            "channel": "p",
                            "valid_from": "2026-01-01T00:00:00Z",
                            "valid_to": "2027-01-01T00:00:00Z",
                            "input_unit": "kPa",
                            "output_unit": "kPa",
                            "slope": "1",
                            "intercept": ".1",
                            "uncertainty": ".01",
                            "minimum": "0",
                            "maximum": "100",
                            "issued_at": "2025-12-01T00:00:00Z",
                        }
                    ],
                    "policies": [
                        {
                            "id": "q1",
                            "unit": "kPa",
                            "lower": "0",
                            "upper": "20",
                            "required_tags": ["lot"],
                        }
                    ],
                    "bindings": [
                        {"instrument": "press-1", "channel": "p", "policy": "q1"}
                    ],
                }
            )
            values = [
                {
                    "id": f"r{i}",
                    "source": "bench",
                    "sequence": i,
                    "instrument": "press-1",
                    "channel": "p",
                    "event_time": f"2026-11-11T00:00:{i * 10:02d}Z",
                    "value": str(10 + i),
                    "unit": "kPa",
                    "uncertainty": "0",
                    "tags": {"lot": "synthetic-A"},
                }
                for i in range(3)
            ]
            rows = read_jsonl("\n".join(json.dumps(value) for value in values))
            ingestion = app.pipeline.ingest(
                "demo-batch", rows, received_at="2026-11-11T00:01:00Z"
            )
            duplicate = app.pipeline.ingest(
                "demo-batch", rows, received_at="2026-11-11T00:01:00Z"
            )
            app.catalog.watermark("bench", "2026-11-11T00:01:00Z")
            window = app.windows.materialize(
                {
                    "id": "minute",
                    "instrument": "press-1",
                    "channel": "p",
                    "start": "2026-11-11T00:00:00Z",
                    "end": "2026-11-11T00:01:00Z",
                    "seconds": 60,
                    "sources": ["bench"],
                }
            )
            app.windows.seal("minute", window["generation"])
            snapshot = app.snapshots.create("release-1", ["minute"])
            app.snapshots.export("release-1", root / "snapshot.json")
            verification = verify_export(root / "snapshot.json")
            integrity = app.inspector.check()
            assert ingestion["accepted"] == 3
            assert duplicate["reused"] is True
            assert window["windows"][0]["statistics"]["mean"] == "11.1"
            assert verification["observations"] == 3
            gauge = _gauge_demonstration(app)
            return {
                "ingestion": ingestion,
                "duplicate_reused": duplicate["reused"],
                "mean": window["windows"][0]["statistics"]["mean"],
                "snapshot": snapshot,
                "export_verified": verification,
                "gauge_study": gauge,
                "integrity": integrity,
            }


def _gauge_demonstration(app):
    """Crossed gauge R&R with three operators, three parts and two trials."""
    parts = ["part-a", "part-b", "part-c"]
    operators = ["op-1", "op-2", "op-3"]
    app.gauge_studies.plan(
        {
            "id": "grr-demo",
            "instrument": "press-1",
            "channel": "p",
            "start": "2026-11-11T00:00:00Z",
            "end": "2026-11-12T00:00:00Z",
            "parts": parts,
            "operators": operators,
            "replicates": 2,
            "sources": ["bench"],
            "tolerance_width": "20",
            "process_variation": "8",
        }
    )
    app.gauge_studies.start("grr-demo", 1)
    part_offset = {"part-a": 0, "part-b": 4, "part-c": 8}
    operator_offset = {"op-1": 0, "op-2": 0.5, "op-3": 1}
    values = []
    sequence = 100
    for part in parts:
        for operator in operators:
            for replicate in (1, 2):
                sequence += 1
                value = (
                    10
                    + part_offset[part]
                    + operator_offset[operator]
                    + (0.1 if replicate == 2 else 0)
                )
                values.append(
                    {
                        "id": f"gr{sequence}",
                        "source": "bench",
                        "sequence": sequence,
                        "instrument": "press-1",
                        "channel": "p",
                        "event_time": f"2026-11-11T01:{sequence % 60:02d}:00Z",
                        "value": str(value),
                        "unit": "kPa",
                        "uncertainty": "0",
                        "tags": {
                            "lot": "synthetic-A",
                            "part": part,
                            "operator": operator,
                            "replicate": str(replicate),
                        },
                    }
                )
    rows = read_jsonl("\n".join(json.dumps(value) for value in values))
    app.pipeline.ingest("grr-demo-batch", rows, received_at="2026-11-12T01:00:00Z")
    for value in values:
        app.gauge_studies.bind("grr-demo", value["id"])
    completeness = app.gauge_studies.completeness("grr-demo")
    assert completeness["balanced"]
    app.gauge_studies.submit("grr-demo", 2)
    decided = app.gauge_studies.decide(
        "grr-demo", 3, "lead-metrologist", True, "Gauge variation within budget"
    )
    report = decided["report"]
    return {
        "state": decided["state"],
        "balanced": completeness["balanced"],
        "bound_observations": completeness["bound_observations"],
        "gauge_percent_vs_tolerance": report["formula_evaluation"][
            "contributions"
        ]["gauge"]["percent_vs_tolerance"],
        "interaction_pooled": report["formula_evaluation"]["interaction"][
            "pooled_into_repeatability"
        ],
    }
